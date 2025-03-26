import cv2
import numpy as np

# Load trajectory data
data = np.loadtxt("target_poses_log.txt")  # Replace with the correct path
x_positions = data[:, 0]
y_positions = data[:, 1]

# Load the video
video_path = "1.mp4"  # Replace with video path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define physical workspace boundaries (adjust for alignment)
x_min, x_max = -0.65, 0.6  # Replace with your actual X-axis range
y_min, y_max = -1.1, -0.4  # Replace with your actual Y-axis range

# Map X, Y positions to pixel coordinates
x_pixels = ((x_positions - x_min) / (x_max - x_min)) * frame_width
y_pixels = frame_height - ((y_positions - y_min) / (y_max - y_min)) * frame_height  # Flip y-axis

# Calculate how many trajectory points to display per frame
points_per_frame = len(x_positions) / num_frames

# ====== Video Writer Setup ======
output_path = "trajectory_overlay.mp4"  # Output video path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video and overlay trajectory
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Determine the number of points to display based on the frame index
    num_points_to_display = int(frame_idx * points_per_frame)

    # Draw trajectory up to the current number of points
    for i in range(num_points_to_display):
        cv2.circle(frame, (int(x_pixels[i]), int(y_pixels[i])), 2, (0, 0, 255), -1)  # Red dots

    # Write the frame with overlay to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Trajectory Visualization", frame)
    frame_idx += 1

    # Exit with 'q'
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Wait time based on video FPS
        break

# Release resources
cap.release()
out.release()  # Save the video
cv2.destroyAllWindows()

print(f"Video saved to: {output_path}")
