#!/usr/bin/env python

# 정해진 경로대로 이동, 초기위치로 돌아오기 (파라미터값 조정 필요, 정확한 골포지션)
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped

# Callbacks definition
def done_cb(status, result):
    if status == 3:
        rospy.loginfo("Goal reached")
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
pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped)
initpose_msg = PoseWithCovarianceStamped()
initpose_msg.header.frame_id = "map"
initpose_msg.pose.pose.position.x = 0.1803995817899704
initpose_msg.pose.pose.position.y = 0.056696951389312744
initpose_msg.pose.pose.orientation.x = 0.0
initpose_msg.pose.pose.orientation.y = 0.0
initpose_msg.pose.pose.orientation.z = 0
initpose_msg.pose.pose.orientation.w = 0.9999999999999988
rospy.sleep(1)
rospy.loginfo("Setting initial pose")
pub.publish(initpose_msg)
rospy.loginfo("Initial pose SET")

# Initializing navigation client
navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
navclient.wait_for_server()

# # # First navigation goal
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

# # Sending first navigation goal
navclient.send_goal(goal1, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# # Second navigation goal
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

# # Sending second navigation goal
navclient.send_goal(goal2, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# # Third navigation goal
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
navclient.send_goal(goal3, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
navclient.wait_for_result()

# # Fourth navigation goal
# goal4 = MoveBaseGoal()
# goal4.target_pose.header.frame_id = "map"
# goal4.target_pose.header.stamp = rospy.Time.now()
# goal4.target_pose.pose.position.x = 0.01924784077285243
# goal4.target_pose.pose.position.y = 0.08436867452749593
# goal4.target_pose.pose.position.z = 0.0
# goal4.target_pose.pose.orientation.x = 0.0
# goal4.target_pose.pose.orientation.y = 0.0
# goal4.target_pose.pose.orientation.z = -0.7021836726803337
# goal4.target_pose.pose.orientation.w = 0.7119958495814129

# # Sending third navigation goal
# navclient.send_goal(goal4, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
# navclient.wait_for_result()

rospy.spin()


