import pandas as pd
import numpy as np
from scipy import constants

from ekf_class import Filter
from ekf_plot import *

# input_file_name = "test_data/imu_test3.csv"
input_file_name = "static_test_medium.csv"
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
output_plot_name = "IMU euler angles.png"
save_states_to_csv = True
plot_states = True


def main():

    try:
        imu_data = pd.read_csv(input_file_name, engine="python")
        imu_data = imu_data.drop(columns_to_drop, axis=1, errors="ignore")

        start_time = imu_data["timestamp [ns]"].iloc[0]
        previous_time = start_time

        # Initialize filter class instance
        # at ca. 100 Hz, 500 rows corresponds to 5 seconds of calibration
        ekf = Filter(sample_amount_for_calibration=100)

        for index, row in imu_data.iterrows():

            # prepare inputs
            current_time = row["timestamp [ns]"]
            delta_t = (current_time - previous_time) / 1e9
            elapsed_time = (current_time - start_time) / 1e9

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
            u_g = gyro * (np.pi / 180.0)
            u_a = accel * constants.g
            # ---

            # run each sample
            if not ekf.is_calibrated:
                ekf.calibration_step(accel, u_g)
                previous_time = current_time
                continue
            ekf.prediction_step(u_g, u_a, delta_t)
            ekf.correction_step(u_g, u_a)
            # ---

            # end of cycle
            ekf.states_history.append(np.append(ekf.x.copy(), elapsed_time))
            previous_time = current_time
            # ---

        # Save states to CSV
        if save_states_to_csv:
            ekf.save_states_to_csv(output_file_name)
            if plot_states:
                pyplot_euler(output_file_name, output_plot_name)
                # pyplot_quaternions(output_file_name, output_plot_name)

        print("---")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file_name}'.")


if __name__ == "__main__":
    main()
