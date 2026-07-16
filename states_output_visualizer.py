import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Read the CSV file
df = pd.read_csv("states_over_time_v2.csv")

# Extract quaternions
quaternions = df[["qx", "qy", "qz", "qw"]].values

# Convert quaternions to Euler angles (yaw, pitch, roll) in degrees
rotations = Rotation.from_quat(quaternions)
euler_angles = rotations.as_euler("xyz", degrees=True)  # yaw, pitch, roll
df["roll"] = euler_angles[:, 1]
df["pitch"] = euler_angles[:, 0]
df["yaw"] = euler_angles[:, 2]

# Time axis: ~9.52ms between samples (1/105Hz)
sample_rate = 105  # Hz
time_step = 1.0 / sample_rate
time = np.arange(len(df)) * time_step

# Plot roll, pitch, yaw
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
