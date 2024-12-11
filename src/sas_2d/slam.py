import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from motion.imu import IMUMeasurement
from sas_2d.sas_2d import SAS
from util.config import Config
from util.util import wrap2pi


class FixedLagSmoother:

    def __init__(self,
                 config: Config,
                 sas: SAS,
                 init_state: np.ndarray,
                 init_pulses: np.ndarray,
                 init_accel: np.ndarray,
                 init_prior: np.ndarray,
                 init_prior_cov: np.ndarray
                 ):

        # SLAM Initialization
        self._sas = sas
        self._pulses = init_pulses
        self._state = init_state
        self._prior = init_prior
        self._accel = init_accel
        self._prior_cov = init_prior_cov

        # Config
        self._c = config
        self._signal_t = self._c.Ts * np.arange(int(self._c.max_range / self._c.Ts))

        # Constants
        self._imu_cov           = np.eye(2) * 1e-2 ** 2
        self._pos_process_cov   = np.eye(2) * 1e-2 ** 2
        self._vel_process_cov   = np.eye(2) * 1e-2 ** 2
        self._phase_var         = 1e-3


    @property
    def state(self):
        return self._state

    @staticmethod
    def importance_sample(img, n_samples):
        weights = img ** 4
        weights = weights / np.sum(weights)
        flat_weights = weights.flatten()
        indices = np.arange(len(flat_weights))

        samples = np.random.choice(indices, size=n_samples, replace=False, p=flat_weights)

        sample_y, sample_x = np.unravel_index(samples, img.shape)

        return np.stack((sample_x, sample_y), axis=-1)

    @staticmethod
    def compute_sample_roundtrip_t(sample_coords, poses):
        sample_vec = sample_coords[np.newaxis, :] - poses[:, np.newaxis]
        sample_range = np.linalg.norm(sample_vec, axis=-1)
        sample_dir = sample_vec / sample_range[:, :, np.newaxis]  # N_poses x N_samples x 2
        sample_rt_t = 2 * sample_range / C

        return sample_rt_t, sample_dir

    @staticmethod
    def build_prior_system(state, prior, cov):
        sqrt_inv_cov = np.linalg.inv(sp.linalg.sqrtm(cov))
        A = sqrt_inv_cov
        b = sqrt_inv_cov @ (prior - state[0])
        return A, b


    @staticmethod
    def build_motion_system(accel, accel_measurement_cov, vel_process_cov, pos_process_cov, dt):
        # accel is (N_poses, 2)

        N_poses = accel.shape[0]

        # d_accel / d_state using position
        H_pos = np.array([
            [-2/dt**2, 0, -2/dt, 0,     2/dt**2, 0, 0, 0],
            [0, -2/dt**2, 0, -2/dt,     0, 2/dt**2, 0, 0],
        ])

        accel_pos_cov = accel_measurement_cov + 1/dt**4 * pos_process_cov
        inv_sqrt_accel_pos_cov = np.linalg.inv(sp.linalg.sqrtm(accel_pos_cov))

        A_pos = inv_sqrt_accel_pos_cov @ H_pos

        # d_accel / d_state using velocity
        H_vel = np.array([
            [0, 0, -1/dt, 0,    0, 0, 1/dt, 0],
            [0, 0, 0, -1/dt,    0, 0, 0, 1/dt],
        ])

        accel_vel_cov = accel_measurement_cov + 1/dt**2 * vel_process_cov
        inv_sqrt_accel_vel_cov = np.linalg.inv(sp.linalg.sqrtm(accel_vel_cov))

        A_vel = inv_sqrt_accel_vel_cov @ H_vel

        A_pose_pair = np.concatenate((A_pos, A_vel), axis=0)  # 4 x 8

        A = np.zeros(((N_poses - 1) * 4, N_poses * 4))
        b = np.zeros(N_poses * 4)

        for pose_idx in range(0, N_poses - 1):
            t = 4 * pose_idx
            l = 4 * pose_idx
            A[t:t+4, l:l+8] = A_pose_pair

            b_pos = inv_sqrt_accel_pos_cov @ accel[pose_idx]
            b_vel = inv_sqrt_accel_vel_cov @ accel[pose_idx]
            b[t:t+4] = np.concatenate((b_pos, b_vel), axis=0)

        return A, b


    @staticmethod
    def build_phase_system(phase_error, sample_dir, sample_weights, phase_var):

        N_poses, N_samples = phase_error.shape

        phase_grad = 2 * K * sample_dir # N_poses x N_samples x 2

        sqrt_inv_var = 1 / np.sqrt(phase_var)

        A = np.zeros((N_poses * N_samples, N_poses * 4))
        for pose_i in range(N_poses):
            l = 4 * pose_i
            t = N_samples * pose_i
            A[t:t+N_samples, l:l+2] = sqrt_inv_var * sample_weights[:, np.newaxis] * phase_grad[pose_i]

        b = sqrt_inv_var * (sample_weights * phase_error).reshape(-1)

        return A, b

    def build_linear_system(self, map_sample_coords, map_samples, dt):

        state = self._state
        accel = self._accel

        # Dimensions
        N_poses = state.shape[0]
        N_samples = map_sample_coords.shape[0]
        assert state.shape[0] == accel.shape[0]

        sample_rt_t, sample_dir = FixedLagSmoother.compute_sample_roundtrip_t(map_sample_coords, state[:, :2])  # N_poses x N_samples

        # Pulse interpolation
        k = sample_rt_t / self._c.Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts
        k_i_plus_1 = np.clip(k_i + 1, 0, self._pulses.shape[1] - 1)  # Upper bounds (clipped)
        row_indices = np.arange(N_poses)[:, np.newaxis]
        row_indices = np.repeat(row_indices, N_samples, axis=1)

        pulses = (1 - k_a) * self._pulses[row_indices, k_i] + k_a * self._pulses[row_indices, k_i_plus_1]

        # update = pulses * np.exp(w * sample_rt_t)  # N_poses x N_samples
        update = pulses * np.exp(1j * self._c.w * sample_rt_t)

        sample_phases = np.angle(map_samples)
        sample_weights = np.abs(map_samples)

        est_phase = np.angle(update)
        phase_error = wrap2pi(est_phase - sample_phases)

        # Measurement Jacobian
        A = np.zeros((N_poses * 4 + N_samples * N_poses, N_poses * 4))
        # A = np.zeros((N_poses * 2 + N_samples * N_poses, N_poses * 4))

        b = np.empty((N_poses * 4 + N_samples * N_poses))
        # b = np.empty((N_poses * 2 + N_samples * N_poses))

        A[:4, :4], b[:4] = FixedLagSmoother.build_prior_system(state, self._prior, self._prior_cov)

        A[4:N_poses * 4, :], b[4:N_poses * 4] = FixedLagSmoother.build_motion_system(accel,
                                                                                     self._imu_cov,
                                                                                     self._vel_process_cov,
                                                                                     self._pos_process_cov,
                                                                                     dt)

        A[-N_poses*N_samples:, :], b[-N_poses*N_samples:] = FixedLagSmoother.build_phase_system(phase_error,
                                                                                                sample_dir,
                                                                                                sample_weights,
                                                                                                self._phase_var)

        return A, b

    def marginalize_and_advance(self, pulse: np.ndarray, accel_measurement: np.ndarray, dt) -> None:
        # marginalize
        self._prior = self._state[1]

        # advance acceleration
        self._accel[:-1] = self._accel[1:]
        self._accel[-1] = accel_measurement

        # advance state
        self._state[:-1] = self._state[1:]
        last_pos, last_vel, last_accel = self._state[-2, :2], self._state[-2, 2:], self._accel[-2]
        next_pos = last_pos + dt * last_vel + 0.5 * last_accel * dt**2
        next_vel = last_vel + dt * last_accel
        self._state[-1] = np.concatenate((next_pos, next_vel), axis=0)

        # add new pulse
        self._pulses[:-1] = self._pulses[1:]
        self._pulses[-1] = pulse

    def update(self, accel_measurement: np.ndarray, pulse: np.ndarray, dt):
        # marginalize out the last pose and shift the state vector
        self.marginalize_and_advance(pulse, accel_measurement, dt)

        norm_map = self._sas.map / np.max(np.abs(self._sas.map))

        # Sample map
        sample_idx = FixedLagSmoother.importance_sample(np.abs(norm_map), 128)
        sample_coords = self._sas.grid_pos[sample_idx[:, 1], sample_idx[:, 0]]
        samples = norm_map[sample_idx[:, 1], sample_idx[:, 0]]

        n_iterations = 10
        for i in range(n_iterations):
            A, b = self.build_linear_system(sample_coords, samples, dt)

            A = csr_matrix(A)

            res = splu(A.T @ A, permc_spec="COLAMD")
            delta = res.solve(A.T @ b)

            delta = delta.reshape((self._lag, 4))
            self._state += delta

        self._sas.update_map(self._state[0, :2], self._pulses[0])
