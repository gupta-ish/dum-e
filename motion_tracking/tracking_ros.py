import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import PointStamped

# Initialize ROS Node
rospy.init_node("pvc_pipe_tracker", anonymous=True)
pose_pub = rospy.Publisher("/pvc_pipe_pose", PointStamped, queue_size=10)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

# moving average
class ObjectTracker:
    def __init__(self, window_size=10):
        self.positions = []
        self.window_size = window_size
        
    def update(self, x, y, z):
        self.positions.append((x, y, z))
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
            
    def get_smoothed_pose(self):
        if not self.positions:
            return None
        return np.mean(self.positions, axis=0)

tracker = ObjectTracker()

while not rospy.is_shutdown():
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Adjusted range for green
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        if valid:
            largest = max(valid, key=cv2.contourArea)

            # Use minAreaRect for more stable centroid
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(color_image, [box], 0, (0, 255, 0), 2)

            cx, cy = int(rect[0][0]), int(rect[0][1])
            depth = depth_frame.get_distance(cx, cy)
            if 0.1 < depth < 2.5:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                tracker.update(*point_3d)

                smoothed = tracker.get_smoothed_pose()
                if smoothed is not None:
                    # Display on image
                    label = f"X:{smoothed[0]:.2f} Y:{smoothed[1]:.2f} Z:{smoothed[2]:.2f}"
                    cv2.putText(color_image, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # ROS message
                    point_msg = PointStamped()
                    point_msg.header.stamp = rospy.Time.now()
                    point_msg.header.frame_id = "camera_link"
                    point_msg.point.x = smoothed[0]
                    point_msg.point.y = smoothed[1]
                    point_msg.point.z = smoothed[2]
                    pose_pub.publish(point_msg)

    # Display
    cv2.imshow("Color", color_image)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
