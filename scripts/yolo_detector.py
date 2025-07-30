#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String  # We'll publish the detected object's name
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import os

# --- IMPORTANT ---
# You must specify the paths to your YOLOv4 configuration files.
# These paths are typically within your catkin workspace or a dedicated models folder.
# The brief mentions a YOLO_COCO_DATA_PATH, which you should use.
# For this example, we assume they are in a 'yolo' folder inside the package.

# Construct the paths relative to the package location
package_path = os.path.join(os.path.dirname(__file__), '..')  # Goes up to the package root
YOLO_CONFIG_PATH = os.path.join(package_path, 'yolo/yolov4.cfg')
YOLO_WEIGHTS_PATH = os.path.join(package_path, 'yolo/yolov4.weights')
YOLO_NAMES_PATH = os.path.join(package_path, 'yolo/coco.names')


class YoloDetector:
    def __init__(self):
        rospy.loginfo("Initializing YOLO Detector Node...")

        # Check if YOLO files exist
        if not all(os.path.exists(p) for p in [YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH, YOLO_NAMES_PATH]):
            rospy.logerr("YOLO config/weights/names file not found. Please check the paths in yolo_detector.py.")
            return

        # Load YOLOv4 network
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        # Load class names
        with open(YOLO_NAMES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            rospy.loginfo("YOLO classes loaded successfully.")

        self.bridge = CvBridge()
        self.detection_publisher = rospy.Publisher('/yolo_detections', String, queue_size=1)

        # The video player node publishes on '/camera/image'
        self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        rospy.loginfo("YOLO Detector is ready and subscribed to camera topic.")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        height, width, _ = cv_image.shape
        blob = cv2.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.15:  # Confidence threshold
                    label = str(self.classes[class_id])
                    # Publish the detected object's name
                    self.detection_publisher.publish(label)
                    rospy.loginfo(f"Detected: {label} with confidence {confidence:.2f}")


def main():
    rospy.init_node('yolo_detector_node')
    try:
        YoloDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An error occurred in yolo_detector: {e}")


if __name__ == '__main__':
    main()
