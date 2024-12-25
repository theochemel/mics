import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

from motion.imu import IMUMeasurement, get_world_accel_2d
from motion.trajectory import Trajectory

files = [
    'results_lag_2_init_2.pkl',
    'results_lag_2_init_5.pkl',
    'results_lag_2_init_10.pkl',
    'results_lag_2_init_40.pkl'
]

dicts = []
for name in files:
    with open(name, 'rb') as f:
        dicts.append(pkl.load(f))

position_errors = []

def dead_reckon(imu_measurements: IMUMeasurement):
    dr_traj = np.zeros((len(imu_measurements.acceleration_body) + 1, 4))
    dr_traj[0] = [4.5, 4.5, 0, 0]

    dt = 0.05

    for i in range(len(imu_measurements.acceleration_body)):
        accel_world = get_world_accel_2d(
            imu_measurements.acceleration_body[i],
            imu_measurements.orientation_rpy[i, 2]
        )
        v = dr_traj[i, 2:]
        dx = dt * v + 0.5 * dt**2 * accel_world
        dv = dt * accel_world

        dr_traj[i+1, :2] = dr_traj[i, :2] + dx
        dr_traj[i+1, 2:] = dr_traj[i, 2:] + dv

    return dr_traj

fig, ax = plt.subplots(figsize=(6,4 ))

labels = [
    '2 Init Poses',
    '5 Init Poses',
    '10 Init Poses',
    '40 Init Poses'
]

for d, label in zip(dicts, labels):
    gt_traj : Trajectory= d['gt_traj']
    traj = d['computed_traj'][:-10]
    dr_traj = dead_reckon(d['imu_measurements'])[:-11]

    pos_error = np.linalg.norm(traj[:, :2] - gt_traj.position_world[:-10, :2], axis=1)
    vel_error = np.linalg.norm(traj[:, 2:] - gt_traj.velocity_world[:-10, :2], axis=1)

    dr_pos_error = np.linalg.norm(dr_traj[:, :2] - gt_traj.position_world[:-10, :2], axis=1)
    dr_vel_error = np.linalg.norm(dr_traj[:, 2:] - gt_traj.velocity_world[:-10, :2], axis=1)

    ax.plot(vel_error, label=label)

ax.plot(dr_vel_error, label='Dead Reckoned')
plt.suptitle('Velocity Error')
plt.xlabel('Pose index')
plt.ylabel('Velocity error (m/s)')
plt.legend(loc='upper right')
plt.show()
fig.savefig('init_vel_errors.svg', dpi=600, bbox_inches='tight')

