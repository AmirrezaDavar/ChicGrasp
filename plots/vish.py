import zarr
import numpy as np
import os

# Define the path to your Zarr store
store_path = 'replay_buffer.zarr'
store = zarr.DirectoryStore(store_path)
root = zarr.open(store)

# List of datasets to download, including 'action'
datasets = ['left_jaw', 'right_jaw', 'robot_eef_pose', 'robot_joint', 'action']

# Define the output directory (pushZ-master)
output_dir = '/home/wanglab/1_REF_ws/git_1_action8/8action'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and save each dataset as a .txt file in the output directory
for dataset in datasets:
    try:
        # Load the data from Zarr
        data = root[f'data/{dataset}'][:]
        
        # Save data to a text file in the output directory
        file_name = f'{dataset}_data.txt'
        file_path = os.path.join(output_dir, file_name)
        np.savetxt(file_path, data, fmt='%f')  # Use '%f' for floating-point numbers
    
        print(f"{dataset} data saved successfully to '{file_path}'")
            
    except KeyError as e:
        print(f"Error loading data for {dataset}: {e}")
