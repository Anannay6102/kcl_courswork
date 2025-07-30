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
from sound_play.libsoundplay import SoundClient

# --- Global Variables & Callbacks ---

request_lock = threading.Lock()
latest_request = None

yolo_detection_lock = threading.Lock()
object_found_event = threading.Event()
target_search_item = ""

ROOM_COORDINATES = {
    'A': (2.0, 8.0, 1.0),  # Pantry
    'B': (6.0, 8.0, 1.0),  # Guest Room
    'C': (10.0, 8.0, 1.0),  # Guest Room
    'D': (2.0, 3.0, 1.0),  # Front Desk
    'E': (6.0, 3.0, 1.0),  # Lobby
    'F': (10.0, 3.0, 1.0)  # Guest Room
}


def request_callback(message):
    global latest_request
    with request_lock:
        rospy.loginfo("--- New Request Received ---")
        rospy.loginfo("Room: %s, Request: %s", message.room, message.request)
        latest_request = message


def yolo_callback(message):
    global target_search_item
    detected_object = message.data.strip()
    with yolo_detection_lock:
        if target_search_item and detected_object == target_search_item:
            rospy.loginfo(f"Matching object '{detected_object}' found! Setting event.")
            object_found_event.set()


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
    def __init__(self, target_room_key=None):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['room_in'])
        self.target_room_key = target_room_key
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for 'move_base' action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("'move_base' action server not available.")
        else:
            rospy.loginfo("'move_base' action server found.")

    def execute(self, userdata):
        room_key = self.target_room_key if self.target_room_key else userdata.room_in
        rospy.loginfo(f"Executing state: NavigateTo Room {room_key}")
        coords = ROOM_COORDINATES.get(room_key)
        if not coords:
            rospy.logerr(f"Unknown room: {room_key}. Aborting navigation.")
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
            rospy.loginfo(f"Successfully navigated to Room {room_key}.")
            return 'succeeded'
        else:
            rospy.logerr(f"Failed to navigate to Room {room_key}.")
            return 'aborted'


class SearchForObject(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],
                             input_keys=['item_in'])
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def execute(self, userdata):
        global target_search_item, object_found_event
        with yolo_detection_lock:
            target_search_item = userdata.item_in.strip()
            object_found_event.clear()
        rospy.loginfo(f"Executing state: SearchForObject. Looking for a '{target_search_item}'.")
        turn_cmd = Twist()
        turn_cmd.angular.z = 0.5
        while not rospy.is_shutdown():
            self.cmd_vel_pub.publish(turn_cmd)
            if object_found_event.wait(timeout=0.1):
                rospy.loginfo(f"SUCCESS: Event received, found '{target_search_item}'!")
                self.cmd_vel_pub.publish(Twist())
                with yolo_detection_lock:
                    target_search_item = ""
                return 'succeeded'
        self.cmd_vel_pub.publish(Twist())
        return 'aborted'


class Speak(smach.State):
    def __init__(self, text_to_speak):
        smach.State.__init__(self, outcomes=['succeeded'], input_keys=['item_in'])
        self.text_template = text_to_speak
        self.sound_client = SoundClient(blocking=True)

    def execute(self, userdata):
        final_text = self.text_template.format(
            item=userdata.item_in) if '{item}' in self.text_template else self.text_template
        rospy.loginfo(f"Executing state: Speak. Saying: '{final_text}'")
        self.sound_client.say(final_text)
        rospy.sleep(1)
        return 'succeeded'


class CheckForPerson(smach.State):
    def __init__(self, search_duration=10.0):
        smach.State.__init__(self, outcomes=['person_found', 'person_not_found', 'aborted'])
        self.search_duration = rospy.Duration(search_duration)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def execute(self, userdata):
        global target_search_item, object_found_event
        rospy.loginfo("Executing state: CheckForPerson. Scanning room for a person.")
        with yolo_detection_lock:
            target_search_item = "person"
            object_found_event.clear()
        start_time = rospy.Time.now()
        turn_cmd = Twist()
        turn_cmd.angular.z = 0.3
        while (rospy.Time.now() - start_time) < self.search_duration:
            if rospy.is_shutdown():
                self.cmd_vel_pub.publish(Twist())
                return 'aborted'
            self.cmd_vel_pub.publish(turn_cmd)
            if object_found_event.wait(timeout=0.1):
                rospy.loginfo("SUCCESS: Person found in the room!")
                self.cmd_vel_pub.publish(Twist())
                with yolo_detection_lock:
                    target_search_item = ""
                return 'person_found'
        rospy.loginfo("FAIL: No person found after scanning for the full duration.")
        self.cmd_vel_pub.publish(Twist())
        with yolo_detection_lock:
            target_search_item = ""
        return 'person_not_found'


def main():
    rospy.init_node('main_node')

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
                               transitions={'succeeded': 'SEARCH_FOR_OBJECT', 'aborted': 'GO_TO_LOBBY'})

        smach.StateMachine.add('SEARCH_FOR_OBJECT', SearchForObject(),
                               transitions={'succeeded': 'ANNOUNCE_OBJECT_FOUND', 'aborted': 'GO_TO_LOBBY'},
                               remapping={'item_in': 'item'})

        smach.StateMachine.add('ANNOUNCE_OBJECT_FOUND', Speak("I have found the {item}"),
                               transitions={'succeeded': 'GO_TO_DELIVERY_ROOM'},
                               remapping={'item_in': 'item'})

        smach.StateMachine.add('GO_TO_DELIVERY_ROOM', NavigateTo(),
                               transitions={'succeeded': 'CHECK_FOR_PERSON', 'aborted': 'GO_TO_FRONT_DESK'},
                               remapping={'room_in': 'room'})

        smach.StateMachine.add('CHECK_FOR_PERSON', CheckForPerson(),
                               transitions={'person_found': 'ANNOUNCE_DELIVERY',
                                            'person_not_found': 'GO_TO_FRONT_DESK',
                                            'aborted': 'GO_TO_LOBBY'})

        smach.StateMachine.add('ANNOUNCE_DELIVERY', Speak("I am here to deliver your {item}"),
                               transitions={'succeeded': 'GO_TO_LOBBY'},  # <-- Updated transition
                               remapping={'item_in': 'item'})

        smach.StateMachine.add('GO_TO_FRONT_DESK', NavigateTo('D'),
                               transitions={'succeeded': 'ANNOUNCE_FAILURE', 'aborted': 'GO_TO_LOBBY'})

        smach.StateMachine.add('ANNOUNCE_FAILURE', Speak("I could not find anyone in the room to deliver the {item}"),
                               transitions={'succeeded': 'GO_TO_LOBBY'},  # <-- Updated transition
                               remapping={'item_in': 'item'})

        # --- Final State to Return to Lobby ---
        smach.StateMachine.add('GO_TO_LOBBY', NavigateTo('E'),
                               transitions={'succeeded': 'WAIT_FOR_REQUEST', 'aborted': 'WAIT_FOR_REQUEST'})

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
