import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
c = 1500  # Speed of sound in water (m/s)
f = 30e3  # Frequency of the sound (Hz)
l = c / f

# Array parameters
Nx = 8  # Number of elements in the x-direction
Ny = 8  # Number of elements in the y-direction
# dx = l / 2  # Element spacing in x-direction
# dy = l / 2  # Element spacing in y-direction
dx = 0.02
dy = 0.02

steering_phi = pi
steering_theta = 0.2

# Angles
Phi = np.linspace(0, 2 * np.pi, 360)  # Azimuth angle
Theta = np.linspace(0, pi / 2, 180)     # Elevation angle
Phi, Theta = np.meshgrid(Phi, Theta)

# x = dx * np.arange(0, Nx)
# y = dy * np.arange(0, Ny)

# x, y = np.meshgrid(x, y)
# x, y = x.flatten(), y.flatten()
N = 32
psi = np.linspace(0, 2 * pi, N)
rs = [0.15, 0.2]
x = np.concatenate([r * np.cos(psi) for r in rs])
y = np.concatenate([r * np.sin(psi) for r in rs])

positions = np.stack((
    x, y, np.zeros_like(x),
), axis=1)


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
    Theta = np.linspace(0, pi / 2, 180)
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

    Phi = np.linspace(0, 2 * np.pi, 360)
    Theta = np.linspace(0, pi / 2, 180)
    Phi, Theta = np.meshgrid(Phi, Theta)

    R = response_grid(Phi, Theta, ks, wavelength, positions)

    Rdb = 20 * np.log10(R)

    plt.imshow(Rdb, vmax=0, vmin=-80)
    plt.colorbar()
    plt.title("Array Response (dB)")
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

    Theta = np.linspace(-pi / 2, pi / 2, 180)

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

    fig.tight_layout()
    plt.show()



def plot_geometry(positions):
    fig, axs = plt.subplots(ncols=2, figsize=(6.4, 3.2))

    min = np.min(positions) - 0.05
    max = np.max(positions) + 0.05

    axs[0].scatter(positions[:, 0], positions[:, 1])
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")
    axs[0].set_xlim([min, max])
    axs[0].set_ylim([min, max])
    axs[0].set_title("XY Plane")

    axs[1].scatter(positions[:, 0], positions[:, 2])
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].set_aspect("equal")
    axs[1].set_xlim([min, max])
    axs[1].set_ylim([min, max])
    axs[1].set_title("XZ Plane")

    fig.suptitle("Array Geometry")
    fig.tight_layout()
    plt.show()


# plot_response_2d(l, l, 0, 0, positions)
# plot_response_3d(l, l, 0, 0, positions)
for steering_theta in np.linspace(0, pi / 4, 4):
    plot_response_slices(l, l, 0, steering_theta, positions)
# plot_geometry(positions)
