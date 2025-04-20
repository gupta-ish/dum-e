import numpy as np
import cv2
import pyrealsense2 as rs
from multiprocessing import shared_memory
import time

####################
shm_name = "object_position_shm"
shm_size = 3 * 4  

try:
    print(f"Creating shared memory {shm_name} of size {shm_size} bytes.")
    time.sleep(2)
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
except FileExistsError:
    print(f"Shared memory {shm_name} already exists. Using existing memory.")
    shm = shared_memory.SharedMemory(name=shm_name, create=False, size=shm_size)

shm_array = np.ndarray((3,), dtype=np.float32, buffer=shm.buf)
shm_array[:] = 0 

#####################

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

pc = rs.pointcloud()

align = rs.align(rs.stream.color)

class ObjectTracker:
    def __init__(self):
        self.positions = []
        
    def update(self, x, y, z):
        self.positions.append((x, y, z))
        if len(self.positions) > 20:
            self.positions.pop(0)
            
    def get_trajectory(self):
        return self.positions

tracker = ObjectTracker()

# shm_c = shared_memory.SharedMemory("", True, 32)

while True:
    frames = pipeline.wait_for_frames()
    
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([40, 40, 40]) # np.array([0, 50, 50])  # Example for red objects
    upper_bound = np.array([85, 255, 255]) #np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        center_x, center_y = x + w//2, y + h//2


        ##############################################################
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            depth = depth_frame.get_distance(cx, cy)
            if 0.1 < depth < 2.5:  # Filter out noisy points (0–2.5m)
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                tracker.update(point_3d[0], point_3d[1], point_3d[2])
                # shm_array[0] = point_3d[2]
                # shm_array[1] = -point_3d[0]
                # shm_array[2] = -point_3d[1]

                shm_array[0] = point_3d[0]
                shm_array[1] = point_3d[1]
                shm_array[2] = point_3d[2]
                
                print(f"shm_array: {shm_array[0]:.2f}, {shm_array[1]:.2f}, {shm_array[2]:.2f}")

                text = f"X: {point_3d[0]:.2f}m Y: {point_3d[1]:.2f}m Z: {point_3d[2]:.2f}m"
                cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                trajectory = tracker.get_trajectory()
                for i in range(1, len(trajectory)):
                    p1 = rs.rs2_project_point_to_pixel(depth_intrin, trajectory[i - 1])
                    p2 = rs.rs2_project_point_to_pixel(depth_intrin, trajectory[i])
                    cv2.line(color_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
        ##############################################################

    cv2.imshow("Color", color_image)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

shm.close()