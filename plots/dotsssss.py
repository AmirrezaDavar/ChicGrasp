import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the background image
image_path = "dot.png"  # Replace with your image file
background = cv2.imread(image_path)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# Step 2: Load the robot action data
data_file = "target_poses_log.txt"  # Replace with your file path
data = np.loadtxt(data_file)  # Load the log file

# Extract X, Y, Z and gripper states
x_positions = data[:, 0]  # X coordinate (meters)
y_positions = data[:, 1]  # Y coordinate (meters)
z_positions = data[:, 2]  # Z coordinate (meters)
left_jaw = data[:, 6]     # Left jaw state
right_jaw = data[:, 7]    # Right jaw state

# Step 3: Scale the X and Y positions to image pixel dimensions
image_height, image_width, _ = background.shape

# Assuming X and Y range within known workspace bounds:
# Replace these bounds with the actual workspace limits (in meters)
x_min, x_max = -0.2, 0.2  # Example X range
y_min, y_max = -0.7, -0.4  # Example Y range

# Map X and Y to image pixel coordinates
x_scaled = np.interp(x_positions, (x_min, x_max), (0, image_width))
y_scaled = np.interp(y_positions, (y_min, y_max), (image_height, 0))  # Flip Y for image

# Step 4: Use Z values for dot size and gripper state for color
z_scaled = np.interp(z_positions, (min(z_positions), max(z_positions)), (20, 100))  # Dot size
gripper_state = (left_jaw + right_jaw) / 2  # Average gripper state
colors = plt.cm.plasma(1 - gripper_state)  # Color mapping: plasma colormap

# Step 5: Plot the image with overlaid positions
plt.figure(figsize=(10, 8))
plt.imshow(background)

# Scatter plot for scaled positions
plt.scatter(x_scaled, y_scaled, c=colors, s=z_scaled, alpha=0.8, edgecolor='k', label="Robot Actions")

# Add labels and legend
plt.title("Corrected Robot Actions Overlay")
plt.axis("off")
plt.legend(loc="upper right")
plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), label="Gripper State (0=Closed, 1=Open)")
plt.show()
