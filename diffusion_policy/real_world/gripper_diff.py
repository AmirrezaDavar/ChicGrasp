import serial
import time
from datetime import datetime
from pynput import keyboard
import threading
from multiprocessing import Value  # Import Value for shared variables

class GripperController:
    # def __init__(self, port='/dev/ttyACM0', baudrate=9600, log_file="gripper_status_log.txt"):
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        # self.log_file = log_file

        # Use multiprocessing.Value for shared variables
        self.left_jaw_state = Value('i', 1)  # Assuming the left jaw is open at the start
        self.right_jaw_state = Value('i', 1)  # Assuming the right jaw is open at the start

        # Set up serial communication with Arduino
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            time.sleep(0.01)  # Give some time for the connection to establish
            print("Serial connection established.")

            # Force the gripper to open both jaws to match our initial assumption.
            self.send_command('open_left_request')
            self.update_state(left=1, action="forced_open_left")
            self.send_command('open_right_request')
            self.update_state(right=1, action="forced_open_right")

        except serial.SerialException:
            print("Error: Could not connect to Arduino.")
            self.ser = None  # Prevent errors if serial is not connected

        # # Log the initial state
        # self.log_gripper_status("initial_state")

    def update_state(self, left=None, right=None, action=None):
        """Update jaw states safely using locks."""
        if left is not None:
            with self.left_jaw_state.get_lock():
                self.left_jaw_state.value = left
        if right is not None:
            with self.right_jaw_state.get_lock():
                self.right_jaw_state.value = right
        # if action:
        #     self.log_gripper_status(action)

    # def log_gripper_status(self, action):
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     left_state = self.left_jaw_state.value
    #     right_state = self.right_jaw_state.value
    #     log_entry = f"Timestamp: {timestamp}, Left_Jaw_State: {left_state}, Right_Jaw_State: {right_state}, Action: {action}\n"
    #     with open(self.log_file, "a") as file:
    #         file.write(log_entry)
        # print(f"Gripper Log -> {log_entry.strip()}")  # Debug Print

    def on_press(self, key_char):
        """
        Adjusted to accept a single character representing the key press.
        """
        try:
            char = key_char  # key_char is already a character

            # Debug statement to verify key press
            print(f"on_press called with key_char: '{char}'")

            if char == 'o':
                self.send_command('open_right_request')
                self.update_state(right=1, action="open_right")
            elif char == 'l':
                self.send_command('close_right_request')
                self.update_state(right=0, action="close_right")
            elif char == 'i':
                self.send_command('open_left_request')
                self.update_state(left=1, action="open_left")
            elif char == 'k':
                self.send_command('close_left_request')
                self.update_state(left=0, action="close_left")
            else:
                print(f"No action assigned for key: '{char}'")
        except AttributeError as e:
            print(f"AttributeError in on_press: {e}")
            pass  # Do nothing if AttributeError occurs

    def send_command(self, command):
        if self.ser:
            print(f"Sending command: {command}")
            self.ser.write(f'{command}\n'.encode())
            self.ser.flush()
            print(f"Command sent: {command}")

    def set_state(self, left_jaw_state, right_jaw_state):
        """Set the gripper state directly, sending commands only if the state changes."""
        left_jaw_state = int(left_jaw_state)
        right_jaw_state = int(right_jaw_state)

        # Flags to check if any state changes
        changed = False

        # Update left jaw if needed
        if left_jaw_state != self.left_jaw_state.value:
            changed = True
            if left_jaw_state == 1:
                self.send_command('open_left_request')
                self.update_state(left=1, action="open_left")
            else:
                self.send_command('close_left_request')
                self.update_state(left=0, action="close_left")

        # Update right jaw if needed
        if right_jaw_state != self.right_jaw_state.value:
            changed = True
            if right_jaw_state == 1:
                self.send_command('open_right_request')
                self.update_state(right=1, action="open_right")
            else:
                self.send_command('close_right_request')
                self.update_state(right=0, action="close_right")

        # # If no change, just log the current state (optional)
        # if not changed:
        #     self.log_gripper_status("no_change_maintain_state")

    def on_release(self, key):
        # Exit program on 'q'
        if key.char == 'q':
            return False

    def start_key_listener(self):
        # Key listener is not required in the updated code
        pass

    def get_states(self):
        """Return the current jaw states safely."""
        with self.left_jaw_state.get_lock(), self.right_jaw_state.get_lock():
            left = self.left_jaw_state.value
            right = self.right_jaw_state.value
            # print(f"Fetching Gripper States -> Left: {left}, Right: {right}")  # Debug Print
            return left, right

    def close(self):
        # Close the serial connection when done
        if self.ser:
            self.ser.close()

