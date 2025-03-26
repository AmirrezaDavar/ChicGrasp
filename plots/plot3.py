import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the data (X, Y positions from the log file)
data = np.loadtxt("target_poses_log.txt")  # Replace with the path to your file
x = data[:, 0]  # X positions
y = data[:, 1]  # Y positions

# Load the top-view image
image_path = "1.png"  # Replace with the path to your image
background = cv2.imread(image_path)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Scale X, Y to fit the image dimensions
# Assuming the X and Y values are normalized or range close to actual workspace
image_height, image_width, _ = background.shape

# Define workspace boundaries (these depend on your robot setup, adjust accordingly)
x_min, x_max = -0.5, 0.5  # X-axis range in meters
y_min, y_max = -0.9, -0.4  # Y-axis range in meters

# Convert real-world X, Y positions to pixel coordinates on the image
x_pixels = ((x - x_min) / (x_max - x_min)) * image_width
y_pixels = ((y - y_min) / (y_max - y_min)) * image_height

# Flip y_pixels since image coordinates start from the top
y_pixels = image_height - y_pixels

# Plot the image and overlay the trajectory
plt.figure(figsize=(10, 8))
plt.imshow(background)
plt.scatter(x_pixels, y_pixels, c='red', s=5, label='Trajectory', alpha=0.7)
plt.title("Robot XY Trajectory Overlaid on Top-View Setup")
plt.axis("off")  # Hide axes
plt.legend(loc="upper right")
plt.show()
