import numpy as np
import matplotlib.pyplot as plt

from motion.imu import IMUMeasurement

def plot_imu_traces(measurement: IMUMeasurement) -> plt.Figure:
    m = measurement

    fig, axs = plt.subplots(2)

    axs[0].set_title("Acceleration")
    axs[0].plot(m.time, m.acceleration_body[:, 0], label="X")
    axs[0].plot(m.time, m.acceleration_body[:, 1], label="Y")
    axs[0].plot(m.time, m.acceleration_body[:, 2], label="Z")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title("Orientation")
    axs[1].plot(m.time, m.orientation_rpy[:, 0], label="Roll")
    axs[1].plot(m.time, m.orientation_rpy[:, 1], label="Pitch")
    axs[1].plot(m.time, m.orientation_rpy[:, 2], label="Yaw")
    axs[1].legend()
    axs[1].grid()

    return fig
