import numpy as np
from math import pi, cos, sin, tan
import matplotlib.pyplot as plt
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
c = 1500  # Speed of sound in water (m/s)
f = 100e3  # Frequency of the sound (Hz)
l = c / f

# Array parameters
Nx = 13  # Number of elements in the x-direction
Ny = 13 # Number of elements in the y-direction
# dx = l / 2  # Element spacing in x-direction
# dy = l / 2  # Element spacing in y-direction
dx = l / 2
dy = l / 2

# print(dx, dy)

# steering_phi = pi
# steering_theta = 0.2

# Angles
# Phi = np.linspace(0, 2 * np.pi, 360)  # Azimuth angle
# Theta = np.linspace(0, pi / 2, 180)     # Elevation angle
# Phi, Theta = np.meshgrid(Phi, Theta)

# x = dx * np.arange(0, Nx)
# y = dy * np.arange(0, Ny)
# x, y = np.meshgrid(x, y)
# x, y = x.flatten(), y.flatten()

# perimeter is 2 * pi * r
# pick base r s.t. 2 * pi * r / N = dx


# Van Trees OAP, 2.15
def direction_vector(phi, theta):
    # returns: \vec{a}

    return np.array([
        -sin(theta) * cos(phi),
        -sin(theta) * sin(phi),
        -cos(theta),
    ])


# Van Trees OAP, 2.28
def array_manifold_vector(wavenumber, positions):
    # wavenumber: \vec{k}
    # positions: [\vec{p}_0, \ldots, \vec{p}_n]
    # returns: \vec{v}_{\vec{k}}{\vec{k}}

    return np.exp(-1j * np.sum(wavenumber * positions, axis=-1))


# Van Trees OAP, 2.24
def wavenumber(wavelength, direction):
    # wavelength: \lambda
    # direction: \vec{a}
    # returns: \vec{k}

    return (2 * pi / wavelength) * direction


# Van Trees OAP, 2.37 and 2.32
def frequency_wavenumber_response(wavenumber, steering_wavenumber, positions):
    N = positions.shape[0]

    k = wavenumber
    ks = steering_wavenumber
    p = positions

    Y = (1 / N) * array_manifold_vector(ks, p).conj().T @ array_manifold_vector(k, p)

    return Y


def response_grid(Phi, Theta, steering_wavenumber, wavelength, positions):
    assert Phi.shape == Theta.shape

    ks = steering_wavenumber

    R = np.zeros(Phi.shape, dtype=np.float64)

    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            phi = Phi[i, j]
            theta = Theta[i, j]

            a = direction_vector(phi, theta)
            k = wavenumber(wavelength, a)

            R[i, j] = abs(frequency_wavenumber_response(k, ks, positions))

    # R = R / np.max(R)

    return R


def response_along_phi(Phi, theta_slice, steering_wavenumber, wavelength, positions):
    Phi = np.array([Phi])
    Theta = np.full_like(Phi, fill_value=theta_slice)

    R = response_grid(Phi, Theta, steering_wavenumber, wavelength, positions)

    return R[0]


def response_along_theta(phi_slice, Theta, steering_wavenumber, wavelength, positions):
    Theta = np.array([Theta])
    Phi = np.full_like(Theta, fill_value=phi_slice)

    R = response_grid(Phi, Theta, steering_wavenumber, wavelength, positions)

    return R[0]


def plot_response_3d(wavelength, steering_wavelength, steering_phi, steering_theta, positions):
    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    Phi = np.linspace(0, 2 * np.pi, 360)
    Theta = np.linspace(-pi / 2, pi / 2, 180)
    Phi, Theta = np.meshgrid(Phi, Theta)

    R = response_grid(Phi, Theta, ks, wavelength, positions)

    Rx = R * np.cos(Phi) * np.sin(Theta)
    Ry = R * np.sin(Phi) * np.sin(Theta)
    Rz = R * np.cos(Theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Rx, Ry, Rz, cmap="viridis")
    ax.axes.set_xlim3d(-1, 1)
    ax.axes.set_ylim3d(-1, 1)
    ax.axes.set_zlim3d(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Array Response (mag)")

    plt.show()


def plot_response_2d(wavelength, steering_wavelength, steering_phi, steering_theta, positions):
    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    Phi = np.linspace(-pi, pi, 360)
    Theta = np.linspace(-pi / 2, pi / 2, 180)
    Phi, Theta = np.meshgrid(Phi, Theta)

    R = response_grid(Phi, Theta, ks, wavelength, positions)

    Rdb = 20 * np.log10(R)

    plt.imshow(Rdb, vmax=0, vmin=-80)
    plt.colorbar()
    plt.title("Array Response (dB)")
    plt.show()


def plot_response_2d_xy(wavelength, fov, steering_wavelength, steering_phi, steering_theta, positions):
    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    z = 1
    extent = z * tan(fov / 2)

    X = np.linspace(-extent, extent, 100)
    Y = np.flip(np.linspace(-extent, extent, 100))
    X, Y = np.meshgrid(X, Y)
    Z = np.full_like(X, fill_value=z)

    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Theta = np.arccos(1 / R)
    Phi = np.arctan2(Y, X)

    R = response_grid(Phi, Theta, ks, wavelength, positions)

    Rdb = 20 * np.log10(R)

    plt.imshow(Rdb, vmax=0, vmin=-80)
    plt.colorbar()
    plt.title(f"Array Response (dB), FOV = {np.rad2deg(fov)} deg")
    plt.show()



def plot_response_slices(wavelength, steering_wavelength, steering_phi, steering_theta, positions):
    fig, axs = plt.subplots(nrows=2)

    Phi = np.linspace(-pi, pi, 360)

    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    R = response_along_phi(Phi, steering_theta, ks, wavelength, positions)

    Rdb = 20 * np.log10(R)

    axs[0].plot(Phi, Rdb)

    axs[0].set_xlabel("Phi (rad)")
    axs[0].set_ylabel("Amplitude (dB)")
    axs[0].set_title("Array Response along Phi")
    axs[0].grid()
    axs[0].axvline(x=steering_phi, c="r")
    axs[0].legend()
    axs[0].set_xlim(-pi, pi)
    axs[0].set_ylim(-80, 10)

    Theta = np.linspace(-pi / 2, pi / 2, 1000)

    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    R = response_along_theta(steering_phi, Theta, ks, wavelength, positions)

    Rdb = 20 * np.log10(R)

    axs[1].plot(Theta, Rdb)

    axs[1].set_xlabel("Theta (rad)")
    axs[1].set_ylabel("Amplitude (dB)")
    axs[1].set_title("Array Response along Theta")
    axs[1].axvline(x=steering_theta, c="r")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_xlim(-pi, pi)
    axs[1].set_ylim(-80, 10)

    fig.tight_layout()
    plt.show()


def compute_beamwidth(wavelength, steering_wavelength, steering_phi, steering_theta, positions):
    Phi = np.linspace(-pi, pi, 10000)
    Theta = np.linspace(-pi / 2, pi / 2, 10000)

    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    R_theta = response_along_theta(steering_phi, Theta, ks, wavelength, positions)
    R_phi = response_along_phi(Phi, steering_theta, ks, wavelength, positions)

    def compute_beamwidth_single(response, Angle, steering_angle):
        Rdb = 20 * np.log10(response)

        steering_angle_index = int(np.where(np.min(np.abs(Angle - steering_angle)) == np.abs(Angle - steering_angle))[0])

        Rdb_left = Rdb[:steering_angle_index]
        Rdb_right = Rdb[steering_angle_index+1:]

        down_three_left_index = np.where(np.min(np.abs(Rdb_left + 3)) == np.abs(Rdb_left + 3))[0]
        down_three_right_index = steering_angle_index + 1 + np.where(np.min(np.abs(Rdb_right + 3)) == np.abs(Rdb_right + 3))[0]

        if len(down_three_left_index) > 1 or len(down_three_right_index) > 1:
            return None

        down_three_left_index = int(down_three_left_index[0])
        down_three_right_index = int(down_three_right_index[0])

        down_three_left_angle = Angle[down_three_left_index]
        down_three_right_angle = Angle[down_three_right_index]

        down_three_beamwidth = down_three_right_angle - down_three_left_angle

        return down_three_beamwidth

    return compute_beamwidth_single(R_phi, Phi, steering_phi), compute_beamwidth_single(R_theta, Theta, steering_theta),


def compute_sidelobe_level(wavelength, steering_wavelength, steering_phi, steering_theta, positions):
    Phi = np.linspace(-pi, pi, 10000)
    Theta = np.linspace(-pi / 2, pi / 2, 10000)

    a_s = direction_vector(steering_phi, steering_theta)
    ks = wavenumber(steering_wavelength, a_s)

    R_theta = response_along_theta(steering_phi, Theta, ks, wavelength, positions)
    R_phi = response_along_phi(Phi, steering_theta, ks, wavelength, positions)

    def compute_sidelobe_level_single(response):
        Rdb = 20 * np.log10(response)

        if np.all(np.isclose(Rdb, 0)):
            return None

        peak_values = Rdb[1:-1][(Rdb[1:-1] > Rdb[0:-2]) & (Rdb[1:-1] > Rdb[2:])]

        second_largest_peak_value = np.partition(peak_values, -2)[-2]

        return float(second_largest_peak_value)

    return compute_sidelobe_level_single(R_phi), compute_sidelobe_level_single(R_theta)


def plot_geometry(positions):
    fig, axs = plt.subplots(ncols=2, figsize=(6.4, 3.2))

    min = np.min(positions) - 0.1 * np.ptp(positions)
    max = np.max(positions) + 0.1 * np.ptp(positions)

    axs[0].scatter(positions[:, 0], positions[:, 1], s=1)
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].set_aspect("equal")
    axs[0].set_xlim([min, max])
    axs[0].set_ylim([min, max])
    axs[0].set_title("XY Plane")

    axs[1].scatter(positions[:, 0], positions[:, 2], s=1)
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Z (m)")
    axs[1].set_aspect("equal")
    axs[1].set_xlim([min, max])
    axs[1].set_ylim([min, max])
    axs[1].set_title("XZ Plane")

    fig.suptitle("Array Geometry")
    fig.tight_layout()
    plt.show()


# plot_response_2d(l, l, 0, 0.0, positions)
# plot_response_2d(l, l, 0, 0.3, positions)

# plot_response_3d(l, l, 0, 0, positions)
#
#
# plot_response_slices(l, l, 0, 0.1, positions)
#
# plot_response_slices(l, l, 0, 0.2, positions)
#
# plot_response_slices(l, l, 0, 0.3, positions)

# for steering_theta in np.linspace(0, pi / 4, 4):
#     plot_response_slices(l, l, 0, steering_theta, positions)

# N = 128

# rs = np.linspace(0.05, 0.2, 20)

spacing = 0.01
# spacing = 0.7 * l
# print(spacing)

n = 15

i = np.arange(0, n)

x = spacing * i + (spacing / 2)
y = spacing * i + (spacing / 2)
# y = np.zeros_like(x)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()

positions = np.stack((
    x, y, np.zeros_like(x),
), axis=1)

diameter = np.ptp(positions)
radius = diameter / 2

center = np.mean(positions, axis=0)

inside_circle = np.linalg.norm(positions - center, axis=1) < 1.01 * radius

positions = positions[inside_circle]

print(positions.shape[0])

plot_geometry(positions)

# plot_response_2d(l, l, 0, 0, positions)
# plot_response_2d(l, l, 0, 0.3, positions)

plot_response_2d_xy(l, pi / 2, l, 0, 0, positions)
plot_response_2d_xy(l, pi / 2, l, 0, 0.3, positions)
plot_response_2d_xy(l, pi / 2, l, 0, 0.6, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 4, 0, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 4, 0.3, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 4, 0.6, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 2, 0, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 2, 0.3, positions)
plot_response_2d_xy(l, pi / 2, l, pi / 2, 0.6, positions)

# plot_response_slices(l, l, 0, 0.3, positions)


# beamwidths = []
# sidelobe_levels = []
#
# fig, ax = plt.subplots()
#
# for n in ns:
#     x = (spacing) * np.arange(0, n)
#     y = (spacing) * np.arange(0, n)
#     x, y = np.meshgrid(x, y)
#     x, y = x.flatten(), y.flatten()
#
#     positions = np.stack((
#         x, y, np.zeros_like(x),
#     ), axis=1)
#
#     a_s = direction_vector(0, 0)
#     ks = wavenumber(l, a_s)
#
#     Theta = np.linspace(-pi / 2, pi / 2, 1000)
#
#     R = response_along_theta(0, Theta, ks, l, positions)
#
#     Rdb = 20 * np.log10(R)
#
#     ax.plot(Theta, Rdb, label=f"{n}")
#
#     _, beamwidth = compute_beamwidth(l, l, 0, 0, positions)
#     _, sidelobe_level = compute_sidelobe_level(l, l, 0, 0, positions)
#
#     beamwidths.append(beamwidth)
#     sidelobe_levels.append(sidelobe_level)
#
# ax.grid()
# # ax.legend()
# ax.set_ylim([-80, 0])
#
# plt.show()
#
# fig, axs = plt.subplots(nrows=2)
#
# axs[0].plot(ns, beamwidths)
# axs[0].grid()
# axs[0].set_xlabel("Grid Size")
# axs[0].set_ylabel("-3db Beamwidth at Normal (rad)")
#
# axs[1].plot(ns, sidelobe_levels)
# axs[1].grid()
# axs[1].set_xlabel("Grid Size")
# axs[1].set_ylabel("Sidelobe Level (dB)")
#
# fig.tight_layout()
#
# plt.show()
#
#
# # ns = [5, 7, 9, 11, 13, 15]
# n = 13
# spacings = np.linspace(l / 8, l, 20)
#
# beamwidths = []
# sidelobe_levels = []
#
# fig, ax = plt.subplots()
#
# for spacing in spacings:
#     x = spacing * np.arange(0, n)
#     y = spacing * np.arange(0, n)
#     x, y = np.meshgrid(x, y)
#     x, y = x.flatten(), y.flatten()
#
#     positions = np.stack((
#         x, y, np.zeros_like(x),
#     ), axis=1)
#
#     a_s = direction_vector(0, 0)
#     ks = wavenumber(l, a_s)
#
#     Theta = np.linspace(-pi / 2, pi / 2, 1000)
#
#     R = response_along_theta(0, Theta, ks, l, positions)
#
#     Rdb = 20 * np.log10(R)
#
#     ax.plot(Theta, Rdb, label=f"{n}")
#
#     _, beamwidth = compute_beamwidth(l, l, 0, 0, positions)
#     _, sidelobe_level = compute_sidelobe_level(l, l, 0, 0, positions)
#
#     beamwidths.append(beamwidth)
#     sidelobe_levels.append(sidelobe_level)
#
# ax.grid()
# # ax.legend()
# ax.set_ylim([-80, 0])
#
# plt.show()
#
# fig, axs = plt.subplots(nrows=2)
#
# axs[0].plot(spacings, beamwidths)
# axs[0].grid()
# axs[0].set_xlabel("Spacing (m)")
# axs[0].set_ylabel("-3db Beamwidth at Normal (rad)")
#
# axs[1].plot(spacings, sidelobe_levels)
# axs[1].grid()
# axs[1].set_xlabel("Spacing (m)")
# axs[1].set_ylabel("Sidelobe Level (dB)")
#
# fig.tight_layout()
#
# plt.show()
