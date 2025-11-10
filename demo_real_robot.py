"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController ####1
from diffusion_policy.real_world.gripper_diff import GripperController
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency

    gripper = GripperController()  # Initialize the GripperController
    # gripper.start_key_listener()   # Start the keyboard listener

    
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                gripper=gripper, # Pass the gripper instance
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            # stop = False
            stop = False
            is_recording = False

            # not sure about this try and finally?
            try:
                while not stop:


                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()
                    # print(obs["robot_eef_pose"])
                    print("Robot 6DOF Pose (robot_eef_pose):", obs['robot_eef_pose'])
                    # print("Robot 6DOF Pose (robot_eef_pose):", obs['left_jaw'])

                    # handle key presses
                    # this is where I am adding gripper control buttons # modified
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            stop = True
                        elif key_stroke == KeyCode(char='c'):
                            # Start recording
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                            key_counter.clear()
                            is_recording = True
                            print('Recording!')
                        elif key_stroke == KeyCode(char='s'):
                            # Stop recording
                            env.end_episode()
                            key_counter.clear()
                            is_recording = False
                            print('Stopped.')

                            # # ---------------------------------------------------
                            # # HARD-CODED PHASE (Step 2) after teleoperation
                            # # ---------------------------------------------------
                            # print("Now moving to hard-coded position...")

                            # waypoints_with_gripper = [
                            #     (np.array([-0.1500000, -0.7400000, 0.09000000, 0.0450000, 3.1300000, 0.0490000]), None),
                            #     (np.array([-0.1500000, -0.7400000, 0.3500000, -0.0000001, -2.1960000, 2.1310000]), None),
                            #     (np.array([-0.1500000, -0.7400000, 0.6250000, -0.0000001, -2.1960000, 2.1310000]), None), # very close to shackle 
                            #     (np.array([-0.1500000, -0.9400000, 0.60565556, -0.0000001, -2.1960000, 2.1310000]), "open"),  # Open gripper
                            #     (np.array([-0.1500000, -0.72657112, 0.60565556, -0.0000001, -2.1960000, 2.1310000]), None),
                            #     (np.array([-0.1500000, -0.7400000, 0.0900000, 0.0450000, 3.1300000, 0.0490000]), None),
                            # ]

                            # # Iterate over waypoints and execute each movement
                            # for i, (waypoint_6d, gripper_action) in enumerate(waypoints_with_gripper):
                            #     print(f"Moving to waypoint {i+1}/{len(waypoints_with_gripper)}: {waypoint_6d}")
                            #     move_timestamp = time.time() + 0.1
                            #     env.exec_actions([waypoint_6d], [move_timestamp], stages=[0])
                                
                            #     print(f"Waiting for robot to reach waypoint {i+1}...")
                            #     time.sleep(5.0)

                            #     # If there's a gripper action, execute it
                            #     if gripper_action == "open":
                            #         print("Opening both jaws...")
                            #         gripper.on_press('o')  # Open right jaw
                            #         gripper.on_press('i')  # Open left jaw
                            #         time.sleep(2.0)  # Wait for gripper to complete action
                            #     elif gripper_action == "close":
                            #         print("Closing both jaws...")
                            #         gripper.on_press('l')  # Close right jaw
                            #         gripper.on_press('k')  # Close left jaw
                            #         time.sleep(2.0)  # Wait for gripper to complete action

                            #     # Optional: Verify if the robot has reached the position
                            #     robot_state = env.get_robot_state()
                            #     actual_pose = robot_state['ActualTCPPose']
                            #     print(f"Robot reached position: {actual_pose}")
                            #     print(f"Target position: {waypoint_6d}")

                            #     # Ensure robot has reached close to the target (optional check)
                            #     if not np.allclose(actual_pose[:3], waypoint_6d[:3], atol=0.01):
                            #         print(f"Warning: Robot did not reach waypoint {i+1} exactly!")

                            # # Update target_pose to the final waypoint to avoid resetting
                            # target_pose = waypoints_with_gripper[-1][0].copy()
                            # print("Hard-coded step complete.") 

                            # # ---------------------------------------------------
                            # # HARD-CODED PHASE (Step 2) after teleoperation
                            # # ---------------------------------------------------                       


                        elif key_stroke == Key.backspace:
                            # Delete the most recent recorded episode
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False
                            # delete


                        # Gripper control based on specific key presses
                        elif key_stroke == KeyCode(char='o'):
                            gripper.on_press('o')  # Open right jaw
                        elif key_stroke == KeyCode(char='l'):
                            gripper.on_press('l')  # Close right jaw
                        elif key_stroke == KeyCode(char='i'):
                            gripper.on_press('i')  # Open left jaw
                        elif key_stroke == KeyCode(char='k'):
                            gripper.on_press('k')  # Close left jaw
 
                    stage = key_counter[Key.space]

                    # visualize
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                    episode_id = env.replay_buffer.n_episodes
                    text = f'Episode: {episode_id}, Stage: {stage}'
                    if is_recording:
                        text += ', Recording!'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(255,255,255)
                    )

                    cv2.imshow('default', vis_img)
                    cv2.pollKey()

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                    # drot_xyz[0] = 0  # Lock roll
                    # drot_xyz[1] = 0  # Lock pitch
                    
                    # uncomment this code to bring back the condition for button chekcs # modified
                    if not sm.is_button_pressed(0):
                        # translation mode
                        drot_xyz[:] = 0
                    else:
                        dpos[:] = 0
                    if not sm.is_button_pressed(1):
                        # 2D translation mode
                        dpos[2] = 0    

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    
                    # target_pose[:3] = np.clip(target_pose[:3], [-0.32, #x limits Right side
                    #                         -0.78, #y limits from the wall
                    #                         -0.0057], #z from the buttom
                    #                         [0.03, #x limits Left side
                    #                             -0.46, #y limits from PC
                    #                             0.05]) #z from the top

                    target_pose[:3] = np.clip(target_pose[:3], [-0.27, #x limits Right side
                                            -0.80, #y limits from the wall
                                            0.0164], #z from the buttom
                                            [0.09, #x limits Left side
                                                -0.50, #y limits from PC
                                                0.05]) #z from the top


                    # Get the current gripper states directly from the GripperController
                    left_jaw_state, right_jaw_state = gripper.get_states()
                    target_action = np.append(target_pose, [left_jaw_state, right_jaw_state])

                    # execute teleop command
                    # Execute teleop command, passing robot_obs to include gripper states
                    env.exec_actions(
                        # actions=[target_pose],
                        actions=[target_action],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        stages=[stage],
                        # robot_obs=robot_obs  # Pass robot_obs here
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

            finally:
                gripper.close()

# %%
if __name__ == '__main__':
    main()