#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32
from hatch.msg import Message 

# Callbacks definition
def done_cb(status, result):
    if status == 3:
        rospy.loginfo("Goal reached")
        if current_goal == 1:
            linear_msg = Int32()
            linear_msg.data = 1900
            linear_pub.publish(linear_msg)
            topic_msg = Message()
            topic_msg.mode = 1
            ttopic_pub.publish(topic_msg)
        elif current_goal == 3:
            topic_msg = Message()
            topic_msg.mode = 2
            ttopic_pub.publish(topic_msg)
            topic_msg = Message()
            topic_msg.mode = 3
            ttopic_pub.publish(topic_msg)
    elif status == 2 or status == 8:
        rospy.loginfo("Goal cancelled")
    elif status == 4:
        rospy.loginfo("Goal aborted")

def active_cb():
    rospy.loginfo("Goal pose being processed")

def feedback_cb(feedback):
    pass  # You can process feedback here if needed

rospy.init_node('set_initial_and_send_goal', anonymous=True)

# Setting initial pose
pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
initpose_msg = PoseWithCovarianceStamped()
initpose_msg.header.frame_id = "map"
initpose_msg.pose.pose.position.x = 0.1803995817899704
initpose_msg.pose.pose.position.y = 0.056696951389312744
initpose_msg.pose.pose.orientation.x = 0.0
initpose_msg.pose.pose.orientation.y = 0.0
initpose_msg.pose.pose.orientation.z = 0
initpose_msg.pose.pose.orientation.w = 0.9999999999999988
# Initializing navigation client
navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
navclient.wait_for_server()

# Define publishers for linear and ttopic
linear_pub = rospy.Publisher('linear', Int32, queue_size=10)
ttopic_pub = rospy.Publisher('topic', Message, queue_size=10)

# First navigation goal
goal1 = MoveBaseGoal()
goal1.target_pose.header.frame_id = "map"
goal1.target_pose.header.stamp = rospy.Time.now()
goal1.target_pose.pose.position.x = 1
goal1.target_pose.pose.position.y = 0
goal1.target_pose.pose.position.z = 0.0
goal1.target_pose.pose.orientation.x = 0.0
goal1.target_pose.pose.orientation.y = 0.0
goal1.target_pose.pose.orientation.z = -0.7
goal1.target_pose.pose.orientation.w = 0.7

# Sending first navigation goal
current_goal = 1
navclient.send_goal(goal1, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# Second navigation goal
goal2 = MoveBaseGoal()
goal2.target_pose.header.frame_id = "map"
goal2.target_pose.header.stamp = rospy.Time.now()
goal2.target_pose.pose.position.x = 1
goal2.target_pose.pose.position.y = -1.3
goal2.target_pose.pose.position.z = 0.0
goal2.target_pose.pose.orientation.x = 0.0
goal2.target_pose.pose.orientation.y = 0.0
goal2.target_pose.pose.orientation.z = -1
goal2.target_pose.pose.orientation.w = 0
# Sending second navigation goal
current_goal = 2
navclient.send_goal(goal2, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# Third navigation goal
goal3 = MoveBaseGoal()
goal3.target_pose.header.frame_id = "map"
goal3.target_pose.header.stamp = rospy.Time.now()
goal3.target_pose.pose.position.x = 0.3
goal3.target_pose.pose.position.y = -1.4
goal3.target_pose.pose.position.z = 0.0
goal3.target_pose.pose.orientation.x = 0.0
goal3.target_pose.pose.orientation.y = 0.0
goal3.target_pose.pose.orientation.z = 0.0
goal3.target_pose.pose.orientation.w = 1
# Sending third navigation goal
current_goal = 3
navclient.send_goal(goal3, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# Fourth navigation goal
goal4 = MoveBaseGoal()
goal4.target_pose.header.frame_id = "map"
goal4.target_pose.header.stamp = rospy.Time.now()
goal4.target_pose.pose.position.x = 0.01924784077285243
goal4.target_pose.pose.position.y = 0.08436867452749593
goal4.target_pose.pose.position.z = 0.0
goal4.target_pose.pose.orientation.x = 0.0
goal4.target_pose.pose.orientation.y = 0.0
goal4.target_pose.pose.orientation.z = -0.7021836726803337
goal4.target_pose.pose.orientation.w = 0.7119958495814129
# Sending fourth navigation goal
current_goal = 4
navclient.send_goal(goal4, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

rospy.spin()