#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped

goal_positions = [
    {
        'x': 0.7579186345204802,
        'y': -0.40806734824191987,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': -0.828575288715858,
        'qw': 0.5598776571086157
    },
    {
        'x': 0.3849080244271294,
        'y': 1.7744349706434361,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': -0.4097996428677951,
        'qw': 0.9121755602434367
    },
    {
        'x': 0.8750092034281083,
        'y': 1.0277894949145323,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.9124767569988788,
        'qw': 0.4091285469590325
    },
    {
        'x': 0.08071552567121923,
        'y': 0.14939131341943943,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.9228055299838844,
        'qw': 0.38526608185922906
    }
]

# Callbacks definition
def done_cb(status, result):
    if status == 3:  # Goal reached
        rospy.loginfo("Goal reached")
        
        if goal_positions.index(position) == 0:
            # 골 포지션 1에 도착한 경우
            topic_msg = Message()
            topic_msg.mode = 1
            topic_pub.publish(topic_msg)

            linear_msg = Int32()
            linear_msg.data = 1900
            linear_pub.publish(linear_msg)

            topic_msg = Message()
            topic_msg.mode = 2
            topic_pub.publish(topic_msg)

            linear_msg = Int32()
            linear_msg.data = 1035
            linear_pub.publish(linear_msg)

            topic_msg = Message()
            topic_msg.mode = 3
            topic_pub.publish(topic_msg)

            rospy.loginfo("Reached Goal Position 1. Performing custom action...")

        elif goal_positions.index(position) == 2:
            # 골 포지션 3에 도착한 경우
            # 180도 회전
            topic_msg = Message()
            topic_msg.mode = 4
            ttopic_pub.publish(topic_msg)

            topic_msg = Message()
            topic_msg.mode = 5
            ttopic_pub.publish(topic_msg)

            topic_msg = Message()
            topic_msg.mode = 6
            ttopic_pub.publish(topic_msg)

            topic_msg = Message()
            topic_msg.mode = 7
            ttopic_pub.publish(topic_msg)
                        
            rospy.loginfo("Reached Goal Position 3. Performing custom action...")
            
    elif status == 2 or status == 8:
        rospy.loginfo("Goal cancelled")
    elif status == 4:
        rospy.loginfo("Goal aborted")

def active_cb():
    rospy.loginfo("Goal pose being processed")

def feedback_cb(feedback):
    pass  # You can process feedback here if needed

# Function to send navigation goal
def send_navigation_goal(client, x, y, z, qx, qy, qz, qw):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = z
    goal.target_pose.pose.orientation.x = qx
    goal.target_pose.pose.orientation.y = qy
    goal.target_pose.pose.orientation.z = qz
    goal.target_pose.pose.orientation.w = qw

    client.send_goal(goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
    client.wait_for_result()

# Function to set initial pose
def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped)
    initpose_msg = PoseWithCovarianceStamped()
    initpose_msg.header.frame_id = "map"
    initpose_msg.pose.pose.position.x = -0.03875144276862845
    initpose_msg.pose.pose.position.y = -0.03837588195035967
    initpose_msg.pose.pose.orientation.x = 0.0
    initpose_msg.pose.pose.orientation.y = 0.0
    initpose_msg.pose.pose.orientation.z = -0.20541739813499252
    initpose_msg.pose.pose.orientation.w = 0.9786744568769791
    rospy.sleep(1)
    rospy.loginfo("Setting initial pose")
    pub.publish(initpose_msg)
    rospy.loginfo("Initial pose SET")

if __name__ == '__main__':
    rospy.init_node('set_initial_and_send_goal', anonymous=True)
    set_initial_pose()
    navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    navclient.wait_for_server()

    while not rospy.is_shutdown():
        for position in goal_positions:
            send_navigation_goal(
                navclient,
                position['x'],
                position['y'],
                position['z'],
                position['qx'],
                position['qy'],
                position['qz'],
                position['qw']
            )


    # linear_pub = rospy.Publisher('linear', Int32, queue_size=10)
    # ttopic_pub = rospy.Publisher('ttopic', Message, queue_size=10)
    # rospy.Subscriber('control_nav', String, control_nav_callback)
    # rospy.Subscriber('app_topic', String, app_topic_callback)

    rospy.spin()

