# import numpy as np
# import matplotlib.pyplot as plt

# # Example data (replace this with your tensor data)
# data = np.array([
#     # [-0.1509, -0.6429, 0.0881, 0.1357, 3.1284, -0.0188, 0.9959, 0.9968],
#     # [-0.1507, -0.6464, 0.0882, 0.1373, 3.1277, -0.0194, 0.9979, 0.9969],
#     # [-0.1499, -0.6505, 0.0890, 0.1367, 3.1282, -0.0200, 0.9968, 0.9956],
#     # Add more data rows
#     # [-0.1355, -0.6768,  0.0788,  0.1345,  3.1264, -0.0139,  0.9628, 0.9695],
#     # [-0.1369, -0.6832,  0.0905,  0.1419,  3.1331, -0.0216,  0.9838,
#     #        0.9771],
#     #      [-0.1324, -0.6915,  0.0988,  0.1368,  3.1287, -0.0237,  0.9731,
#     #        0.9622],
#     #      [-0.1280, -0.6976,  0.0912,  0.1451,  3.1197, -0.0157,  0.9895,
#     #        0.9671],
#     #      [-0.1242, -0.7052,  0.0861,  0.1398,  3.1309, -0.0194,  0.9758,
#     #        0.9687],
#     #      [-0.1232, -0.7122,  0.0833,  0.1439,  3.1299, -0.0198,  0.9869,
#     #        0.9611],
#     #      [-0.1198, -0.7171,  0.0884,  0.1393,  3.1352, -0.0217,  0.9837,
#     #        0.9681],
#     #      [-0.1171, -0.7206,  0.0901,  0.1341,  3.1279, -0.0156,  0.9831,
#     #        0.9716],
#     #      [-0.1149, -0.7230,  0.0862,  0.1505,  3.1253, -0.0120,  0.9808,
#     #        0.9738],
#     #      [-0.1157, -0.7284,  0.0839,  0.1372,  3.1292, -0.0186,  0.9859,
#     #        0.9735],
#     #      [-0.1109, -0.7331,  0.0835,  0.1437,  3.1244, -0.0202,  0.9851,
#     #        0.9704],
#     #      [-0.1091, -0.7355,  0.0915,  0.1366,  3.1252, -0.0177,  0.9909,
#     #        0.9774],
#     #      [-0.1067, -0.7412,  0.0812,  0.1457,  3.1301, -0.0168,  0.9767,
#     #        0.9511],
#     #      [-0.1020, -0.7435,  0.0844,  0.1418,  3.1298, -0.0203,  0.9837,
#     #        0.9666],
#     #      [-0.0967, -0.7471,  0.0845,  0.1325,  3.1285, -0.0238,  0.9812,
#     #        0.9482]
#     # [-0.14781606, -0.67199314, 0.08875876, 0.13707593, 3.128159, -0.01918637, 1.0, 1.0],
#     # [-0.14680159, -0.6757569, 0.08876876, 0.13740489, 3.1281476, -0.01944371, 1.0, 1.0],
#     # [-0.14554559, -0.6793766, 0.08865427, 0.13663837, 3.1282911, -0.01970334, 1.0, 1.0],
#     # [-0.14435133, -0.6826727, 0.08843812, 0.13699444, 3.1284876, -0.01922158, 1.0, 1.0],
#     # [-0.14333184, -0.6855755, 0.08902588, 0.13709399, 3.1283596, -0.01937371, 1.0, 1.0],
#     # [-0.14254375, -0.6883674, 0.08835249, 0.13674831, 3.1282628, -0.01941224, 1.0, 1.0],
#     # [-0.14145014, -0.6907877, 0.08822387, 0.13653277, 3.1283488, -0.019145, 1.0, 1.0],
#     # [-0.14063272, -0.6932308, 0.08767858, 0.1369243, 3.1283195, -0.01902673, 1.0, 1.0],
#     # [-0.13994327, -0.69544685, 0.08757984, 0.13738494, 3.1284506, -0.0196139, 1.0, 1.0],
#     # [-0.13944815, -0.69774616, 0.08844902, 0.13701212, 3.1281202, -0.01879514, 1.0, 1.0],
#     # [-0.1390756, -0.700039, 0.08860657, 0.13732529, 3.12816, -0.01982237, 1.0, 1.0],
#     # [-0.138146, -0.7026613, 0.08864943, 0.13642305, 3.1280034, -0.01910608, 1.0, 1.0],

    
# ])

# # Columns: x, y, z, roll, pitch, yaw, left_gripper, right_gripper
# x, y, z, roll, pitch, yaw, left_gripper, right_gripper = data.T

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.plot(x, y, z, label='Robot Trajectory')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# plt.title('3D Trajectory of Robot')
# plt.legend()
# plt.show()

# time_steps = range(len(left_gripper))  # Assuming each row is one timestep

# plt.figure(figsize=(10, 5))
# plt.step(time_steps, left_gripper, label='Left Gripper', where='mid')
# plt.step(time_steps, right_gripper, label='Right Gripper', where='mid')
# plt.xlabel('Time Step')
# plt.ylabel('Gripper State (0=Closed, 1=Open)')
# plt.title('Gripper States Over Time')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(time_steps, roll, label='Roll')
# plt.plot(time_steps, pitch, label='Pitch')
# plt.plot(time_steps, yaw, label='Yaw')
# plt.xlabel('Time Step')
# plt.ylabel('Orientation (Radians)')
# plt.title('Orientation Over Time')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load data from the file
data = np.loadtxt('target_poses_log.txt')

# Assuming the data columns correspond to:
# [x, y, z, roll, pitch, yaw, left_gripper, right_gripper]
x, y, z = data[:, 0], data[:, 1], data[:, 2]
roll, pitch, yaw = data[:, 3], data[:, 4], data[:, 5]
left_gripper, right_gripper = data[:, 6], data[:, 7]

# Time steps (assuming each row is one time step)
time_steps = range(len(data))

# Step 2: Plot 3D Trajectory
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, label='Robot Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
plt.title('3D Trajectory of Robot')
plt.legend()
plt.show()



# Step 3: Plot Gripper States Over Time
plt.figure(figsize=(10, 5))
plt.step(time_steps, left_gripper, label='Left Gripper', where='mid')
plt.step(time_steps, right_gripper, label='Right Gripper', where='mid')
plt.xlabel('Time Step')
plt.ylabel('Gripper State (0=Closed, 1=Open)')
plt.title('Gripper States Over Time')
plt.legend()
plt.show()

# Step 4: Plot Orientation Changes Over Time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, roll, label='Roll')
plt.plot(time_steps, pitch, label='Pitch')
plt.plot(time_steps, yaw, label='Yaw')
plt.xlabel('Time Step')
plt.ylabel('Orientation (Radians)')
plt.title('Orientation Over Time')
plt.legend()
plt.show()


