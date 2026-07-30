import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# to create a plot grid:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html


def pyplot_euler(output_file_name, output_plot_name):
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

    plt.figure(figsize=(15, 9))
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


def pyplot_quaternions(output_file_name, output_plot_name):
    df = pd.read_csv(output_file_name)

    time = df["time"].values

    print("---")
    print(f"Plotting quaternion values from {output_file_name}")

    plt.figure(figsize=(15, 9))
    plt.plot(time, df["qx"], label="qx", color="blue")
    plt.plot(time, df["qy"], label="qy", color="orange")
    plt.plot(time, df["qz"], label="qz", color="green")
    plt.plot(time, df["qw"], label="qw", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("quat value")
    plt.title("Quaternion values")
    plt.legend()
    plt.grid(True)
    # plt.savefig(output_plot_name)
    plt.show()


def pyplot_grid(output_file_name, output_plot_name):
    df = pd.read_csv(output_file_name)

    time = df["time"].values

    # convert quaternions to euler angles
    quaternions = df[["qx", "qy", "qz", "qw"]].values
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler("xyz", degrees=True)  # yaw, pitch, roll
    df["roll"] = euler_angles[:, 1]
    df["pitch"] = euler_angles[:, 0]
    df["yaw"] = euler_angles[:, 2]

    fig, axs = plt.subplots(2, 2, figsize=(18, 9))

    # axs[0, 0].set_title("orientation")
    axs[0, 0].plot(time, df["roll"], label="Roll", color="blue")
    axs[0, 0].plot(time, df["pitch"], label="Pitch", color="orange")
    axs[0, 0].plot(time, df["yaw"], label="Yaw", color="green")
    axs[0, 0].set_ylabel("orientation angle (degrees)")

    # axs[0, 1].set_title("velocity")
    axs[0, 1].plot(time, df["vx"], label="vx", color="red")
    axs[0, 1].plot(time, df["vy"], label="vy", color="green")
    axs[0, 1].plot(time, df["vz"], label="vz", color="blue")
    axs[0, 1].set_ylabel("velocity (m/s)")

    # axs[1, 0].set_title("gyro bias")
    axs[1, 0].plot(time, df["b_x"], label="bx", color="red")
    axs[1, 0].plot(time, df["b_y"], label="by", color="green")
    axs[1, 0].plot(time, df["b_z"], label="bz", color="blue")
    axs[1, 0].set_ylabel("gyro bias (m/s)")
    axs[1, 0].set_xlabel("time (s)")

    # axs[1, 1].set_title("position")
    axs[1, 1].plot(time, df["x"], label="px", color="red")
    axs[1, 1].plot(time, df["y"], label="py", color="green")
    axs[1, 1].plot(time, df["z"], label="pz", color="blue")
    axs[1, 1].set_ylabel("position (m)")
    axs[1, 1].set_xlabel("time (s)")

    fig.tight_layout()

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    plt.show()
