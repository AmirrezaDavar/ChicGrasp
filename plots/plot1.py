import matplotlib.pyplot as plt
import numpy as np

# Load data
data = np.loadtxt("target_poses_log.txt")
x = data[:, 0]
y = data[:, 1]

# Plot X-Y trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.title("Robot End-Effector Trajectory (X-Y Plane)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

z = data[:, 2]  # Z positions

# 3D plot of the trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, marker='o', linestyle='-', color='r')

# Labels and title
ax.set_title("3D Trajectory of End-Effector")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
plt.show()

time = np.arange(len(z))  # Create a time axis

plt.figure(figsize=(8, 6))
plt.plot(time, z, color='g', marker='o')
plt.title("Z Position Over Time")
plt.xlabel("Time Step")
plt.ylabel("Z Position (m)")
plt.grid()
plt.show()

left_jaw = data[:, 6]
right_jaw = data[:, 7]

# Plot gripper states over time
plt.figure(figsize=(8, 6))
plt.plot(time, left_jaw, label="Left Jaw", color='orange')
plt.plot(time, right_jaw, label="Right Jaw", color='purple', linestyle='--')
plt.title("Gripper States Over Time")
plt.xlabel("Time Step")
plt.ylabel("Gripper State (0 = Closed, 1 = Open)")
plt.legend()
plt.grid()
plt.show()

import seaborn as sns

# Create a heatmap of X-Y positions
plt.figure(figsize=(8, 6))
sns.kdeplot(x, y, cmap="Blues", fill=True)
plt.title("Heatmap of Robot End-Effector Positions (X-Y Plane)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(z, left_jaw, color='red', label="Left Jaw")
plt.scatter(z, right_jaw, color='blue', label="Right Jaw")
plt.title("Correlation Between Z Position and Gripper State")
plt.xlabel("Z Position (m)")
plt.ylabel("Gripper State (0 = Closed, 1 = Open)")
plt.legend()
plt.grid()
plt.show()


roll = data[:, 3]
pitch = data[:, 4]
yaw = data[:, 5]

plt.figure(figsize=(8, 6))
plt.plot(time, roll, label="Roll", color='r')
plt.plot(time, pitch, label="Pitch", color='g')
plt.plot(time, yaw, label="Yaw", color='b')
plt.title("Orientation (Roll, Pitch, Yaw) Over Time")
plt.xlabel("Time Step")
plt.ylabel("Rotation (Radians)")
plt.legend()
plt.grid()
plt.show()
