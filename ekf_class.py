import numpy as np
from scipy import constants
import pandas as pd
import dill

with open("observer_functions_and_matrices.pkl", "rb") as f:
    ekf_funcs = dill.load(f)


class Filter:

    def __init__(self, sample_amount_for_calibration=200):
        self.calibration_samples = sample_amount_for_calibration
        self.is_calibrated = False
        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

        self.x = None
        self.P = None
        self.Q = None
        self.R = None

        self.states_history = []

    def calibration_step(self, accel, u_g):
        self.accel_calibration_buffer.append(accel)
        self.gyro_calibration_buffer.append(u_g)

        # last calibration step
        if len(self.accel_calibration_buffer) >= self.calibration_samples:
            self._initialize_filter()

    def _initialize_filter(self):
        self._initialize_state_x()
        self._initialize_state_covariance_matrix_P()
        self._initialize_process_noise_covariance_matrix_Q()
        self._initialize_measurement_noise_covariance_matrix_R()

        # Free up memory
        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []
        self.is_calibrated = True

    def _initialize_state_x(self):
        accel_array = np.array(self.accel_calibration_buffer)
        avg_accel = np.mean(accel_array, axis=0)

        gyro_array = np.array(self.gyro_calibration_buffer)
        avg_gyro = np.mean(gyro_array, axis=0)

        measured_gravity = avg_accel / np.linalg.norm(avg_accel)
        global_gravity = np.array([0.0, 0.0, 1.0])

        qw = 1.0 + np.dot(measured_gravity, global_gravity)
        qx, qy, qz = np.cross(measured_gravity, global_gravity)

        q_initial = self._normalize_quat(np.array([qw, qx, qy, qz]))

        self.x = np.array(
            [
                *q_initial,  # orientation
                *np.zeros(6),  # velocity and position
                *avg_gyro,  # gyro bias
            ]
        )

    def _initialize_state_covariance_matrix_P(self):
        # decent certainty for the orientation at the start, high certainty (even smaller number) for the velocity and position
        orientation_certainty = 0.1
        vel_pos_certainty = 1e-6  # Almost fully confident in the velocity and position since I define them at the start as ground truth
        gyro_bias_certainty = 1e-3
        state_certainty = np.concatenate(
            (
                np.repeat(orientation_certainty, 4),
                np.repeat(vel_pos_certainty, 6),
                np.repeat(gyro_bias_certainty, 3),
            )
        )
        self.P = np.diag(state_certainty)

    def _initialize_process_noise_covariance_matrix_Q(self):
        orientation_noise = 1e-8
        velocity_noise = 1e-4
        position_noise = 1e-5
        gyro_bias_noise = 1e-10
        process_noise = np.concatenate(
            (
                np.repeat(orientation_noise, 4),
                np.repeat(velocity_noise, 3),
                np.repeat(position_noise, 3),
                np.repeat(gyro_bias_noise, 3),
            )
        )
        self.Q = np.diag(process_noise)

    def _initialize_measurement_noise_covariance_matrix_R(self):
        accelerometer_noise = 1e-2
        self.R = np.diag(np.repeat(accelerometer_noise, 3))

    def prediction_step(self, u_g, u_a, dt):
        f_q = ekf_funcs["f_q"](
            *self.x[0:4],
            *self.x[10:13],
            *u_g,
            dt,
        ).flatten()
        f_a = ekf_funcs["f_a"](
            *self.x[0:4],
            *u_a,
            constants.g,
        ).flatten()
        A = ekf_funcs["A"](
            *self.x[0:13],
            *u_a,
            *u_g,
            dt,
            constants.g,
        )

        # update orientation
        self.x[0:4] = self._normalize_quat(f_q)

        # update position and velocity
        v_current = self.x[4:7]
        p_current = self.x[7:10]

        x_v_new = v_current + (f_a * dt)
        x_p_new = p_current + (x_v_new * dt)

        self.x[4:7] = x_v_new
        self.x[7:10] = x_p_new

        # update state covariance matrix (currently no noise w so W is just the identity matrix)
        # P = APA.T + WQW.T
        self.P = A @ self.P @ A.T + self.Q

    def correction_step(self, u_g, u_a):
        H = ekf_funcs["H"](*self.x[0:4], constants.g)

        # K = PH(HPH.T + VRV.T)^(-1)
        # change R if the IMU is static
        accel_is_static = np.absolute(np.linalg.norm(u_a) - constants.g) < 0.1
        gyro_is_static = np.linalg.norm(u_g) < 0.1
        if accel_is_static and gyro_is_static:
            R = self.R
        else:
            R = np.eye(3) * 1e5

        kalman_gain = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        # x = x + K(y - h(x, v))
        h = ekf_funcs["h"](
            self.x[0], self.x[1], self.x[2], self.x[3], constants.g
        ).flatten()
        self.x = self.x + kalman_gain @ (u_a - h)

        self.x[0:4] = self._normalize_quat(self.x[0:4])

        # P = (I - KH)P
        self.P = (np.eye(13) - kalman_gain @ H) @ self.P

    def save_states_to_csv(self, filename):
        if not self.states_history:
            print("No states to save.")
            return

        columns = [
            "qw",
            "qx",
            "qy",
            "qz",
            "vx",
            "vy",
            "vz",
            "x",
            "y",
            "z",
            "b_x",
            "b_y",
            "b_z",
            "time",
        ]
        df = pd.DataFrame(self.states_history, columns=columns)
        df.to_csv(filename, index=False)

        print("---")
        print(f"Saved state history to csv: {filename}")
        return

    @staticmethod
    def _normalize_quat(q):
        return q / np.linalg.norm(q)
