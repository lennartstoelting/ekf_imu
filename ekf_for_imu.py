import pandas as pd
import numpy as np
from scipy import constants


def main():
    file_name = "imu_test3.csv"
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

    try:
        imu_data = pd.read_csv(file_name, engine="python")
        imu_data = imu_data.drop(columns_to_drop, axis=1, errors="ignore")

        imu_filter = Filter(
            sample_amount_for_calibration=500
        )  # Initialize filter instance (500 rows is roughly 5 seconds of data, sampled at somewhere between 100 and 110Hz)

        previous_time = imu_data["timestamp [ns]"].iloc[0]
        print("\n---")
        print("Starting real-time simulation...")

        for index, row in imu_data.iterrows():
            current_time = row["timestamp [ns]"]

            gyro_sample = np.array(
                [row["gyro x [deg/s]"], row["gyro y [deg/s]"], row["gyro z [deg/s]"]]
            )
            accel_sample = np.array(
                [
                    row["acceleration x [g]"],
                    row["acceleration y [g]"],
                    row["acceleration z [g]"],
                ]
            )

            output = imu_filter.process_new_sample(
                gyro_sample, accel_sample, previous_time, current_time
            )

            previous_time = current_time

        # Save states to CSV
        imu_filter.save_states_to_csv("states_over_time.csv")
        print(f"\nStates saved to 'states_over_time.csv'.")
        print(f"\nFinal state: {imu_filter.state_x}")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_name}'.")


class Filter:
    def __init__(self, sample_amount_for_calibration=200):
        self.cal_total_samples = sample_amount_for_calibration
        self.is_calibrated = False
        self.accel_calibration_buffer = []
        self.state_x = None
        self.states_history = []

    # one row of data at a time, like a real-time loop
    def process_new_sample(self, gyro, accel, previous_time_ns, current_time_ns):

        if not self.is_calibrated:
            self.accel_calibration_buffer.append(accel)
            if len(self.accel_calibration_buffer) >= self.cal_total_samples:
                self._finalize_calibration()

            return None

        dt = (current_time_ns - previous_time_ns) / 1e9

        self._update_velocity_and_position(dt, accel)
        self._update_orientation(dt, gyro)
        
        self.states_history.append(self.state_x.copy())
        # EKF for later
        # return self.state_x
        return "EKF running..."

    def save_states_to_csv(self, filename):
        if not self.states_history:
            print("No states to save.")
            return

        # Convert states_history to a DataFrame
        columns = [
            "qw", "qx", "qy", "qz",  # Orientation quaternion
            "vx", "vy", "vz",        # Velocities
            "x", "y", "z"            # Positions
        ]
        df = pd.DataFrame(self.states_history, columns=columns)
        df.to_csv(filename, index=False)


    def _finalize_calibration(self):
        print("\n---")
        print("Calibration completed:")
        accel_array = np.array(self.accel_calibration_buffer)

        avg_accel = np.mean(accel_array, axis=0)
        print(f"\nAverage Accel Gravity Vector: {avg_accel}")

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
        print(f"Orientation Quaternion: {self.state_x[0:4]}")

        # Free up memory by clearing buffers
        self.accel_calibration_buffer = []
        self.is_calibrated = True

    def _update_velocity_and_position(self, dt, accel):
        # unit conversion
        accel_m_ss = np.array([accel[0], accel[1], accel[2]]) * constants.g

        accel_global = np.array(self._rotate_vector(accel_m_ss, self.state_x[0:4]))
        accel_global = accel_global - np.array([0.0, 0.0, constants.g])

        v_current = self.state_x[4:7]
        p_current = self.state_x[7:10]

        v_new = v_current + (accel_global * dt)
        p_new = p_current + (v_new * dt)

        self.state_x[4:7] = v_new
        self.state_x[7:10] = p_new

        return

    def _update_orientation(self, dt, gyro):
        # unit conversion
        gyro_rad_s = gyro * (np.pi / 180.0)
        dqx, dqy, dqz = 0.5 * gyro_rad_s * dt

        w_new, x_new, y_new, z_new = self._quaternion_mult(
            self.state_x[0:4], [1.0, dqx, dqy, dqz]
        )

        mag = np.sqrt(w_new**2 + x_new**2 + y_new**2 + z_new**2)
        self.state_x[0:4] = np.array([w_new, x_new, y_new, z_new]) / mag

        return

    def _rotate_vector(self, v, q):
        v_quat = [0, v[0], v[1], v[2]]
        q_conj = [q[0], -q[1], -q[2], -q[3]]

        v_rotated = self._quaternion_mult(self._quaternion_mult(q, v_quat), q_conj)

        return [v_rotated[1], v_rotated[2], v_rotated[3]]

    @staticmethod
    def _quaternion_mult(q, r):
        w = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]
        x = q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]
        y = q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1]
        z = q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]

        return [w, x, y, z]


if __name__ == "__main__":
    main()
