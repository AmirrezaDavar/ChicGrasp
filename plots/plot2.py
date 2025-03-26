import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load data from file
data = np.loadtxt("target_poses_log.txt")  # Assuming all 8 dimensions
x, y, z = data[:, 0], data[:, 1], data[:, 2]  # Position
left_jaw, right_jaw = data[:, 6], data[:, 7]  # Gripper states

# Normalize gripper state for marker color
gripper_avg = (left_jaw + right_jaw) / 2  # Average gripper state
gripper_colors = plt.cm.coolwarm(gripper_avg)  # Map to colormap

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Robot End-Effector Animation")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

# Initialize plot elements
line, = ax.plot([], [], [], 'gray', lw=2)  # Trajectory line
point = ax.scatter([], [], [], c='red', s=60, label="EEF Position")

# Set axis limits (update based on your workspace dimensions)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# Update function for animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])

    # Update the scatter point for current position
    point._offsets3d = (x[frame:frame+1], y[frame:frame+1], z[frame:frame+1])
    point.set_color(gripper_colors[frame])

    return line, point

# Create animation
ani = FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)

# Save or display animation
# Uncomment to save (requires ffmpeg): ani.save('robot_motion.mp4', writer='ffmpeg')
plt.legend()
plt.show()
