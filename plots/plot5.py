import cv2
import numpy as np

# ====== Step 1: Load Trajectory Data and Compute Workspace Boundaries ======
# Load trajectory data
data = np.loadtxt("target_poses_log.txt")  # Replace with your actual file path

# Extract X and Y positions
x_positions = data[:, 0]  # X values
y_positions = data[:, 1]  # Y values

# Automatically compute workspace boundaries
x_min, x_max = x_positions.min(), x_positions.max()
y_min, y_max = y_positions.min(), y_positions.max()

# Add optional padding to avoid clipping (e.g., 5 cm padding)
padding = 0.02  # Smaller padding for precise alignment
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

# Print computed boundaries
print(f"Precise Workspace Boundaries:")
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")

# ====== Step 2: Load Video ======
video_path = "1.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ====== Step 3: Map Real-World Coordinates to Pixel Space ======
# Map X, Y positions to pixel coordinates
x_pixels = ((x_positions - x_min) / (x_max - x_min)) * frame_width
y_pixels = frame_height - ((y_positions - y_min) / (y_max - y_min)) * frame_height  # Flip y-axis

# Scale down the dot size based on video resolution
dot_radius = int(min(frame_width, frame_height) * 0.005)  # Scales with video size

# ====== Step 4: Process Video and Overlay Trajectory ======
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Determine the number of points to display based on the frame index
    num_points_to_display = int((frame_idx / num_frames) * len(x_positions))

    # Draw trajectory up to the current number of points
    for i in range(num_points_to_display):
        cv2.circle(frame, (int(x_pixels[i]), int(y_pixels[i])), dot_radius, (0, 0, 255), -1)  # Red dots

    # Display the frame
    cv2.imshow("Trajectory Visualization", frame)
    frame_idx += 1

    # Wait for FPS duration and exit with 'q'
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break

# ====== Step 5: Release Resources ======
cap.release()
cv2.destroyAllWindows()
