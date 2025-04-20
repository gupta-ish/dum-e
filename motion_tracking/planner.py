# from multiprocessing import shared_memory
# import numpy as np


# shm = shared_memory.SharedMemory(name="object_position_shm")
# human_sword_centroid = np.ndarray((3,), dtype=np.float32, buffer=shm.buf)

# print("Current position:", human_sword_centroid[:])


# camera_to_robot_transform = np.array([
#     [0.0, 0.0, 1.0, 0.0],
#     [1.0, 0.0, 0.0, 0.0],
#     [0.0, -1.0, 0.0, 0.0],
#     [0.5, 1.5, -2.5, 1.0]
# ], dtype=np.float32)

# robot_to_world_transform = np.array([ 7.07106781e-01, -7.07106781e-01, -6.12323400e-17,  3.06890567e-01],
#  [-7.07106781e-01, -7.07106781e-01, -3.14018492e-16, -2.36797046e-16],
#  [ 1.78746802e-16,  2.65342408e-16, -1.00000000e+00, 5.90282052e-01],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00])

# transformed_sword_position = human_sword_centroid @ camera_to_robot_transform @ robot_to_world_transform



import time
import numpy as np
from multiprocessing import shared_memory
from autolab_core import RigidTransform
from frankapy import FrankaArm

# Setup shared memory
shm_name = "object_position_shm"
shm = shared_memory.SharedMemory(name=shm_name)
if shm is None: 
    raise RuntimeError(f"Shared memory {shm_name} not found.")
print(f"Using existing shared memory {shm_name} of size {shm.size} bytes.")
object_pos = np.ndarray((3,), dtype=np.float32, buffer=shm.buf)

# Transformation matrices
# camera_to_robot = np.array([
#     [0.0, 0.0, 1.0, 0.0],
#     [1.0, 0.0, 0.0, 0.0],
#     [0.0, -1.0, 0.0, 0.0],
#     [0.5, 1.5, -2.5, 1.0]
# ], dtype=np.float32)

# robot_to_world = np.array([
#     [ 7.07106781e-01, -7.07106781e-01, -6.12323400e-17,  3.06890567e-01],
#     [-7.07106781e-01, -7.07106781e-01, -3.14018492e-16, -2.36797046e-16],
#     [ 1.78746802e-16,  2.65342408e-16, -1.00000000e+00,  5.90282052e-01],
#     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
# ])

# Final transform: camera -> robot -> world
# full_transform = robot_to_world @ camera_to_robot

# Constants
MIN_POS = np.array([-0.35, -0.4, 0.25])
MAX_POS = np.array([0.65,  0.4, 0.9])
SCALE = 1  # Optional: scale if needed


OFFSETS = np.array([0.0, 0.05, 0.05])  

# Fixed rotation for the end-effector
# fixed_rot = np.array([
#     [-0.09799141,  0.08748266,  0.99133469],
#     [-0.70612802, -0.70804639, -0.00731609],
#     [ 0.70127092, -0.70072611,  0.13115642]
# ])

# Initialize robot
fa = FrankaArm(with_gripper=False)
print("Moving to home first...")
fa.reset_joints()

# Main planner loop
print("Starting real-time following loop...")
rate = 1  # Hz
dt = 1.0 / rate

while True:
    try:
        raw_pos = object_pos.copy()


        new_pos = np.array([raw_pos[0], raw_pos[1], raw_pos[2]])

        # Append 1 to make it homogeneous
        pos_h = np.append(new_pos, 1.0)

        # Transform point into world frame
        # world_pos_h = full_transform @ pos_h
        # pos_h[:3] += OFFSETS
        world_pos = pos_h[:3] * SCALE


        # Clip to workspace limits
        clipped_pos = np.clip(world_pos, MIN_POS, MAX_POS)

        # Form the target pose
        pose = RigidTransform(
            # rotation=fixed_rot,
            translation=clipped_pos,
            from_frame='franka_tool',
            to_frame='world'
        )

        print(f"Sword position: {raw_pos}, Clipped position: {clipped_pos}")

        # Send to robot
        fa.goto_pose(pose, duration=dt, use_impedance=True)
        time.sleep(dt)

    except KeyboardInterrupt:
        print("\n🛑 Exiting planner loop.")
        break
shm.close()