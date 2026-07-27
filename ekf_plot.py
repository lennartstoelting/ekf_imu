import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def pyplot_states(output_file_name, output_plot_name):
    df = pd.read_csv(output_file_name)

    time = df["time"].values

    # convert quaternions to euler angles
    quaternions = df[["qx", "qy", "qz", "qw"]].values
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler("xyz", degrees=True)  # yaw, pitch, roll
    df["roll"] = euler_angles[:, 1]
    df["pitch"] = euler_angles[:, 0]
    df["yaw"] = euler_angles[:, 2]

    print("---")
    print(f"Plotting pitch, yaw and roll from {output_file_name}")

    plt.figure(figsize=(10, 6))
    plt.plot(time, df["roll"], label="Roll", color="blue")
    plt.plot(time, df["pitch"], label="Pitch", color="orange")
    plt.plot(time, df["yaw"], label="Yaw", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title("Yaw, Pitch, Roll")
    plt.legend()
    plt.grid(True)
    # plt.savefig(output_plot_name)
    plt.show()
