#!/usr/bin/env python

import rospy
import smach
import smach_ros
import threading
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from resit_coursework.msg import HotelRequest

# --- Global Variables & Callbacks ---

request_lock = threading.Lock()
latest_request = None
yolo_detection_lock = threading.Lock()
last_detection = None

# Using calibrated coordinates from the previous step
ROOM_COORDINATES = {
    'A': (2.0, 8.0, 1.0),  # Pantry
    'B': (6.0, 8.0, 1.0),  # Guest Room
    'C': (10.0, 8.0, 1.0),  # Guest Room
    'D': (2.0, 3.0, 1.0),  # Front Desk
    'E': (6.0, 3.0, 1.0),  # Lobby
    'F': (10.0, 3.0, 1.0)
}

def request_callback(message):
    global latest_request
    with request_lock:
        rospy.loginfo("--- New Request Received ---")
        rospy.loginfo("Room: %s, Request: %s", message.room, message.request)
        latest_request = message


def yolo_callback(message):
    """Callback to store the latest YOLO detection."""
    global last_detection
    with yolo_detection_lock:
        last_detection = message.data


# --- SMACH States ---

class WaitForRequest(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['request_received', 'aborted'],
                             output_keys=['room_out', 'item_out'])

    def execute(self, userdata):
        global latest_request
        rospy.loginfo('Executing state: WaitForRequest. The robot is idle in the lobby.')
        with request_lock:
            latest_request = None
        while not latest_request and not rospy.is_shutdown():
            rospy.sleep(1)
        if rospy.is_shutdown(): return 'aborted'
        with request_lock:
            userdata.room_out = latest_request.room
            userdata.item_out = latest_request.request
        return 'request_received'


class NavigateTo(smach.State):
    def __init__(self, target_room_name):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.target_room_name = target_room_name
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for 'move_base' action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("'move_base' action server not available.")
        else:
            rospy.loginfo("'move_base' action server found.")

    def execute(self, userdata):
        if not self.move_base_client.wait_for_server(rospy.Duration(1.0)):
            rospy.logerr("'move_base' action server not available. Aborting navigation.")
            return 'aborted'
        rospy.loginfo(f"Executing state: NavigateTo {self.target_room_name}")
        coords = ROOM_COORDINATES.get(self.target_room_name)
        if not coords:
            rospy.logerr(f"Unknown room: {self.target_room_name}. Aborting navigation.")
            return 'aborted'
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = coords[0]
        goal.target_pose.pose.position.y = coords[1]
        goal.target_pose.pose.orientation.w = coords[2]
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"Successfully navigated to {self.target_room_name}.")
            return 'succeeded'
        else:
            rospy.logerr(f"Failed to navigate to {self.target_room_name}.")
            return 'aborted'


class SearchForObject(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],
                             input_keys=['item_in'])
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(10)  # 10hz

    def execute(self, userdata):
        global last_detection
        rospy.loginfo(f"Executing state: SearchForObject. Looking for a '{userdata.item_in}'.")

        # Clear any old detections
        with yolo_detection_lock:
            last_detection = None

        # Create a rotation command
        turn_cmd = Twist()
        turn_cmd.angular.z = 0.5  # radians per second

        # Loop until the object is found or ROS is shut down
        while not rospy.is_shutdown():
            # Publish the turn command to make the robot rotate
            self.cmd_vel_pub.publish(turn_cmd)

            with yolo_detection_lock:
                if last_detection and last_detection == userdata.item_in:
                    rospy.loginfo(f"SUCCESS: Found '{userdata.item_in}'!")
                    # Stop the robot
                    self.cmd_vel_pub.publish(Twist())  # Empty Twist stops the robot
                    return 'succeeded'

            self.rate.sleep()

        # This part is reached if rospy is shutdown
        return 'aborted'


def main():
    rospy.init_node('main_node')

    # Subscribe to the necessary topics
    rospy.Subscriber('/hotel_request', HotelRequest, request_callback)
    rospy.Subscriber('/yolo_detections', String, yolo_callback)

    sm = smach.StateMachine(outcomes=['succeeded', 'aborted', 'preempted'])
    sm.userdata.room = ''
    sm.userdata.item = ''

    with sm:
        smach.StateMachine.add('WAIT_FOR_REQUEST', WaitForRequest(),
                               transitions={'request_received': 'GO_TO_PANTRY', 'aborted': 'aborted'},
                               remapping={'room_out': 'room', 'item_out': 'item'})

        smach.StateMachine.add('GO_TO_PANTRY', NavigateTo('A'),
                               transitions={'succeeded': 'SEARCH_FOR_OBJECT', 'aborted': 'WAIT_FOR_REQUEST'})

        smach.StateMachine.add('SEARCH_FOR_OBJECT', SearchForObject(),
                               transitions={'succeeded': 'WAIT_FOR_REQUEST',  # For now, loop back
                                            'aborted': 'WAIT_FOR_REQUEST'},
                               remapping={'item_in': 'item'})

    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
