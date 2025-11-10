###### one plot #######

# import zarr
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load Zarr store
# store_path = 'replay_buffer.zarr'
# store = zarr.DirectoryStore(store_path)
# root = zarr.open(store)

# # Load datasets
# try:
#     left_jaw = root['data/left_jaw'][:]
#     right_jaw = root['data/right_jaw'][:]
#     timestamps = root['data/timestamp'][:]
# except KeyError as e:
#     print(f"Error loading data: {e}")
#     raise SystemExit(e)

# print("Left Jaw shape:", left_jaw.shape)
# print("Right Jaw shape:", right_jaw.shape)
# print("Timestamps shape:", timestamps.shape)

# # Create DataFrame by accessing the first column
# df = pd.DataFrame({
#     'timestamps': timestamps,
#     'left_jaw': left_jaw[:, 0],
#     'right_jaw': right_jaw[:, 0]
# })

# # Plot Left Jaw and Right Jaw over time
# plt.figure(figsize=(12, 6))
# plt.plot(df['timestamps'], df['left_jaw'], label='Left Jaw')
# plt.plot(df['timestamps'], df['right_jaw'], label='Right Jaw')
# plt.title('Left and Right Jaw Positions over Time')
# plt.xlabel('Time')
# plt.ylabel('Jaw Position')
# plt.legend()
# plt.grid(True)
# plt.show()

#### two plots #####

# import zarr
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load Zarr store
# store_path = 'replay_buffer.zarr'
# store = zarr.DirectoryStore(store_path)
# root = zarr.open(store)

# # Load datasets
# try:
#     left_jaw = root['data/left_jaw'][:]
#     right_jaw = root['data/right_jaw'][:]
#     timestamps = root['data/timestamp'][:]
# except KeyError as e:
#     print(f"Error loading data: {e}")
#     raise SystemExit(e)

# print("Left Jaw shape:", left_jaw.shape)
# print("Right Jaw shape:", right_jaw.shape)
# print("Timestamps shape:", timestamps.shape)

# # Create DataFrame by accessing the first column
# df = pd.DataFrame({
#     'timestamps': timestamps,
#     'left_jaw': left_jaw[:, 0],
#     'right_jaw': right_jaw[:, 0]
# })

# # Plot Left Jaw over time
# plt.figure(figsize=(12, 6))
# plt.plot(df['timestamps'], df['left_jaw'], label='Left Jaw', color='blue')
# plt.title('Left Jaw Position over Time')
# plt.xlabel('Time')
# plt.ylabel('Left Jaw Position')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Right Jaw over time
# plt.figure(figsize=(12, 6))
# plt.plot(df['timestamps'], df['right_jaw'], label='Right Jaw', color='red')
# plt.title('Right Jaw Position over Time')
# plt.xlabel('Time')
# plt.ylabel('Right Jaw Position')
# plt.legend()
# plt.grid(True)
# plt.show()

import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Zarr store
store_path = 'replay_buffer.zarr'
store = zarr.DirectoryStore(store_path)
root = zarr.open(store)

# Load datasets
try:
    left_jaw = root['data/left_jaw'][:]
    right_jaw = root['data/right_jaw'][:]
    timestamps = root['data/timestamp'][:]
except KeyError as e:
    print(f"Error loading data: {e}")
    raise SystemExit(e)

print("Left Jaw shape:", left_jaw.shape)
print("Right Jaw shape:", right_jaw.shape)
print("Timestamps shape:", timestamps.shape)

# Create DataFrame by accessing the first column
df = pd.DataFrame({
    'timestamps': timestamps,
    'left_jaw': left_jaw[:, 0],
    'right_jaw': right_jaw[:, 0]
})

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Left Jaw over time
axes[0].plot(df['timestamps'], df['left_jaw'], label='Left Jaw', color='blue')
axes[0].set_title('Left Jaw Position over Time')
axes[0].set_ylabel('Left Jaw Position')
axes[0].legend()
axes[0].grid(True)

# Plot Right Jaw over time
axes[1].plot(df['timestamps'], df['right_jaw'], label='Right Jaw', color='red')
axes[1].set_title('Right Jaw Position over Time')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Right Jaw Position')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()
