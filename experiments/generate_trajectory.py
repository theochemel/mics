import csv
import numpy as np

from spatialmath import SO3


def wrap2pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def generate_circle_coordinates(center, target, radius, velocity, duration, sample_rate, output_file):
    """
    Generate 3D coordinates of a robot moving in a circle and save them to a CSV file.

    Parameters:
    - center: Tuple (x, y, z) for the center of the circle.
    - radius: Radius of the circle (meters).
    - velocity: Linear velocity of the robot (meters per second).
    - duration: Total duration of the movement (seconds).
    - sample_rate: Sampling rate (samples per second).
    - output_file: Output CSV file path.
    """

    # Unpack the center coordinates
    cx, cy, cz = center
    tx, ty, tz = target

    # Calculate the circle's angular velocity (rad/s)
    angular_velocity = velocity / radius

    # Calculate the number of samples
    num_samples = int(duration * sample_rate)

    # Time vector
    timestamps = np.linspace(0, duration, num_samples)

    # Prepare the CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Generate coordinates for each timestamp
        for t in timestamps:
            # Calculate the angle (theta) for the current time
            theta = angular_velocity * t

            # Calculate the x and y coordinates on the circle
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = cz  # Constant z-coordinate

            world_t_vehicle = SO3.RPY(0, 0, np.arctan2(ty - y, tx - x)) @ SO3.TwoVectors(x="x", y="-y")

            roll, pitch, yaw = world_t_vehicle.rpy()

            # Write the row to the CSV file
            writer.writerow([t, x, y, z, roll, pitch, yaw])

    print(f"Data saved to {output_file}")

# Example usage
target_point = (0.0, 0.0, 0.0)
center_point = (0.0, 0.0, 0.0)  # Center of the circle at (x, y, z)
radius = 10.0  # Radius of the circle in meters
velocity = 0.1  # Linear velocity in meters per second
# duration = 2*np.pi*radius / velocity  # Duration of the movement in seconds
duration = 1000
sample_rate = 1  # Sampling rate in samples per second
output_csv = "circular_path.csv"

generate_circle_coordinates(center_point, target_point, radius, velocity, duration, sample_rate, output_csv)
