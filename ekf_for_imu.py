import pandas as pd
import numpy as np
from scipy import constants
import dill

# ---

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
save_to_csv = False

with open("observer_functions_and_matrices.pkl", "rb") as f:
    ekf_funcs = dill.load(f)

# ---


def main():

    try:
        imu_data = pd.read_csv(input_file_name, engine="python")
        imu_data = imu_data.drop(columns_to_drop, axis=1, errors="ignore")

        # Initialize filter instance (500 rows is roughly 5 seconds of data, sampled at somewhere between 100 and 110Hz)
        ekf = Filter(sample_amount_for_calibration=500)

        previous_time = imu_data["timestamp [ns]"].iloc[0]
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

            ekf.process_step(gyro, accel, previous_time, current_time)

            previous_time = current_time

        # Save states to CSV
        if save_to_csv:
            ekf.save_states_to_csv(output_file_name)
            print("---")
            print(f"Saved state history to csv: {output_file_name}")

        print("---\n")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file_name}'.")


class Filter:

    def __init__(self, sample_amount_for_calibration=200):
        self.cal_total_samples = sample_amount_for_calibration
        self.is_calibrated = False
        self.accel_calibration_buffer = []
        self.state_x = None
        self.states_history = []

    # one row of data at a time, like a real-time loop
    def process_step(self, gyro, accel, previous_time_ns, current_time_ns):

        # calibration
        if not self.is_calibrated:
            self.accel_calibration_buffer.append(accel)
            if len(self.accel_calibration_buffer) >= self.cal_total_samples:
                self._finalize_calibration()

            return None

        # setup variables and functions
        dt = (current_time_ns - previous_time_ns) / 1e9
        u_g = gyro * (np.pi / 180.0)
        u_a = accel * constants.g

        f_q = ekf_funcs["f_q"]
        f_a = ekf_funcs["f_a"]

        # update orientation
        x_q_new = f_q(
            self.state_x[0],
            self.state_x[1],
            self.state_x[2],
            self.state_x[3],
            u_g[0],
            u_g[1],
            u_g[2],
            dt,
        ).flatten()
        x_q_normalized = x_q_new / np.linalg.norm(x_q_new)
        self.state_x[0:4] = x_q_normalized

        # update position and velocity
        accel_rotated = f_a(
            self.state_x[0],
            self.state_x[1],
            self.state_x[2],
            self.state_x[3],
            u_a[0],
            u_a[1],
            u_a[2],
            constants.g,
        ).flatten()
        v_current = self.state_x[4:7]
        p_current = self.state_x[7:10]

        v_new = v_current + (accel_rotated * dt)
        p_new = p_current + (v_new * dt)

        self.state_x[4:7] = v_new
        self.state_x[7:10] = p_new
        print(f"new implementation: {[v_new] + [p_new]}")

        # if save_to_csv:
        self.states_history.append(self.state_x.copy())

        return

    def _finalize_calibration(self):
        print("---")
        print("Calibration completed:")
        accel_array = np.array(self.accel_calibration_buffer)

        avg_accel = np.mean(accel_array, axis=0)
        print(f"-> Average Accel Gravity Vector: {avg_accel}")

        measured_gravity = avg_accel / np.linalg.norm(avg_accel)
        global_gravity = np.array([0.0, 0.0, 1.0])

        qw = 1.0 + np.dot(measured_gravity, global_gravity)
        qx, qy, qz = np.cross(measured_gravity, global_gravity)

        # 5. Normalize it to make it a valid rotation quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm

        # Initialize state
        self.state_x = np.array(
            [
                qw,
                qx,
                qy,
                qz,  # Orientation quaternion
                0.0,
                0.0,
                0.0,  # Initial velocities (vx, vy, vz)
                0.0,
                0.0,
                0.0,  # Initial positions  (x, y, z)
            ]
        )
        print(f"-> Orientation Quaternion: {self.state_x[0:4]}")

        # Free up memory by clearing buffers
        self.accel_calibration_buffer = []
        self.is_calibrated = True

    def save_states_to_csv(self, filename):
        if not self.states_history:
            print("No states to save.")
            return

        # Convert states_history to a DataFrame
        columns = [
            "qw",
            "qx",
            "qy",
            "qz",  # Orientation quaternion
            "vx",
            "vy",
            "vz",  # Velocities
            "x",
            "y",
            "z",  # Positions
        ]
        df = pd.DataFrame(self.states_history, columns=columns)
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
