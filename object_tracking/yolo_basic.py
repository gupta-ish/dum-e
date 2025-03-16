import pyrealsense2 as rs
import numpy as np
import cv2
import time

yolo_path = "/home/ameria/MRSD/16-662 Robot Autonomy/Autonomy_Project/RobotAutonomy---SwordFightingRobot/object_tracking"
labelsPath = f"{yolo_path}/coco.names"
weightsPath = f"{yolo_path}/yolov3.weights"
configPath = f"{yolo_path}/yolov3.cfg"

LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting RealSense camera...")
pipeline.start(config)

try:
    while True:
        start_time = time.time()

        # Wait for frames and get color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to OpenCV format
        frame = np.asanyarray(color_frame.get_data())

        # Prepare YOLO input
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Process YOLO detections
        boxes, confidences, classIDs = [], [], []
        H, W = frame.shape[:2]

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:  # Confidence threshold
                    box = detection[:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply Non-Maximum Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # Draw bounding boxes
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display FPS and output frame
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("RealSense YOLO", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    print("[INFO] Stopping RealSense camera...")
    pipeline.stop()
    cv2.destroyAllWindows()