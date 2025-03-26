import matplotlib.pyplot as plt
import numpy as np

# Parse the data into a structured format
data = """
# Paste your data here as a multi-line string
"""

# Clean and format the data
data_lines = data.strip().split("\n")
parsed_data = []

for line in data_lines:
    # Remove brackets and split the line
    line = line.replace("[", "").replace("]", "")
    values = [float(x) for x in line.split()]
    parsed_data.append(values)

# Convert to numpy array for easy handling
data_array = np.array(parsed_data)

# Check the shape of the array
print(f"Data shape: {data_array.shape}")

# Time steps (x-axis)
time_steps = np.arange(data_array.shape[0])

# Plot each dimension (column)
plt.figure(figsize=(12, 8))
for i in range(data_array.shape[1]):
    plt.plot(time_steps, data_array[:, i], label=f'Feature {i+1}')

# Add labels and title
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Visualization of Multi-Dimensional Data')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
