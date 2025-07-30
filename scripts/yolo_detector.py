#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import os

# --- IMPORTANT ---
# You are currently using the full YOLOv4 model. For better performance on a CPU,
# it is highly recommended to switch to the 'yolov4-tiny' model by changing
# the file paths below and ensuring the tiny files are in your 'yolo' folder.

package_path = os.path.join(os.path.dirname(__file__), '..')
YOLO_CONFIG_PATH = os.path.join(package_path, 'yolo/yolov4.cfg')
YOLO_WEIGHTS_PATH = os.path.join(package_path, 'yolo/yolov4.weights')
YOLO_NAMES_PATH = os.path.join(package_path, 'yolo/coco.names')


class YoloDetector:
    def __init__(self):
        rospy.loginfo("Initializing YOLO Detector Node...")

        if not all(os.path.exists(p) for p in [YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH, YOLO_NAMES_PATH]):
            rospy.logerr("YOLO config/weights/names file not found. Please check the paths in yolo_detector.py.")
            return

        self.net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        with open(YOLO_NAMES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            rospy.loginfo("YOLO classes loaded successfully.")

        self.bridge = CvBridge()
        self.detection_publisher = rospy.Publisher('/yolo_detections', String, queue_size=1)

        # *** FIX: Subscribing to the correct topic '/camera/image' ***
        self.image_subscriber = rospy.Subscriber('/camera/image', Image, self.image_callback)

        rospy.loginfo("YOLO Detector is ready and subscribed to the correct camera topic.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        height, width, _ = cv_image.shape
        blob = cv2.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.75:  # Adjusted confidence for better detection
                    label = str(self.classes[class_id])
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
