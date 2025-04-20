import numpy as np
import time
from autolab_core import RigidTransform
from frankapy import FrankaArm

# Initialize Franka
# fa = FrankaArm()
fa = FrankaArm(with_gripper = False)
fa.reset_joints()

# Fixed rotation matrix from getpose()
rotation = np.array([
    [-0.09799141,  0.08748266,  0.99133469],
    [-0.70612802, -0.70804639, -0.00731609],
    [ 0.70127092, -0.70072611,  0.13115642]
])

# Trajectory parameters
z_fixed = 0.45           # Fixed z-height
x_start, x_end = 0.3, 0.5
num_points = 100
amplitude = 0.05         # Sine wave amplitude in y
frequency = 2 * np.pi    # One full wave
print('initialised')
# Generate trajectory points
x_vals = np.linspace(x_start, x_end, num_points)
y_vals = amplitude * np.sin(frequency * (x_vals - x_start) / (x_end - x_start))
z_vals = np.full_like(x_vals, z_fixed)
print('generated trajectory points')
# Create list of poses
poses = [
    RigidTransform(
        # rotation=rotation,
        translation=np.array([x, y, z]),
        from_frame='franka_tool',
        to_frame='world'
    )
    for x, y, z in zip(x_vals, y_vals, z_vals)
]

# Move to start of trajectory
fa.goto_pose(poses[0],duration=5,buffer_time=10, use_impedance=True)
print('moved to start of trajectory')
# Execute trajectory
# for pose in poses:
#     fa.goto_pose(pose, duration=0.01, use_impedance=True)
#     time.sleep(0.01)
#     print('moving to next pose')
