import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import constants

from ekf_class import Filter

input_file_name = "test_data/imu_test3.csv"
columns_to_drop = [
    "recording id",
    "roll [deg]",
    "pitch [deg]",
    "yaw [deg]",
    "quaternion x",
    "quaternion y",
    "quaternion z",
    "quaternion w",
]

output_file_name = "states_over_time_v2.csv"
save_states_to_csv = True
plot_states = True


def main():

    try:
        imu_data = pd.read_csv(input_file_name, engine="python")
        imu_data = imu_data.drop(columns_to_drop, axis=1, errors="ignore")

        # Initialize filter instance (500 rows is roughly 5 seconds of data, sampled at somewhere between 100 and 110Hz)
        ekf = Filter(sample_amount_for_calibration=500)

        start_time = imu_data["timestamp [ns]"].iloc[0]
        previous_time = start_time
        print("\n---")
        print("Starting real-time simulation...")

        for index, row in imu_data.iterrows():
            current_time = row["timestamp [ns]"]

            gyro = np.array(
                [row["gyro x [deg/s]"], row["gyro y [deg/s]"], row["gyro z [deg/s]"]]
            )
            accel = np.array(
                [
                    row["acceleration x [g]"],
                    row["acceleration y [g]"],
                    row["acceleration z [g]"],
                ]
            )

            if not ekf.is_calibrated:
                ekf.calibration_step(accel)
                previous_time = current_time
                continue

            # setup variables and functions
            dt = (current_time - previous_time) / 1e9
            elapsed_time = (current_time - start_time) / 1e9

            u_g = gyro * (np.pi / 180.0)
            u_a = accel * constants.g

            ekf.prediction_step(u_g, u_a, dt)
            ekf.correction_step(u_g, u_a)

            if save_states_to_csv:
                ekf.states_history.append(np.append(ekf.x.copy(), elapsed_time))

            previous_time = current_time

        # Save states to CSV
        if save_states_to_csv:
            ekf.save_states_to_csv(output_file_name)
            print("---")
            print(f"Saved state history to csv: {output_file_name}")

        if save_states_to_csv and plot_states:
            _plot_states()

        print("---\n")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file_name}'.")


def _plot_states():
    df = pd.read_csv(output_file_name)

    quaternions = df[["qx", "qy", "qz", "qw"]].values

    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler("xyz", degrees=True)  # yaw, pitch, roll
    df["roll"] = euler_angles[:, 1]
    df["pitch"] = euler_angles[:, 0]
    df["yaw"] = euler_angles[:, 2]

    time = df["time"].values

    print("---")
    print(f"Plotting converted pitch, yaw and roll from {output_file_name}")

    plt.figure(figsize=(10, 6))
    plt.plot(time, df["roll"], label="Roll", color="blue")
    plt.plot(time, df["pitch"], label="Pitch", color="orange")
    plt.plot(time, df["yaw"], label="Yaw", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title("Yaw, Pitch, Roll")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
