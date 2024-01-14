#!/usr/bin/env python

# 목표위치 지정 - 목표위치까지 이동  

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

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
    pass  



rospy.init_node('send_goal')

navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
navclient.wait_for_server()

goal = MoveBaseGoal()
goal.target_pose.header.frame_id = "map"
goal.target_pose.header.stamp = rospy.Time.now()

goal.target_pose.pose.position.x = 0.8814135352111224
goal.target_pose.pose.position.y = -0.06493596768820997
goal.target_pose.pose.position.z = 0.0
goal.target_pose.pose.orientation.x = 0.0
goal.target_pose.pose.orientation.y = 0.0
goal.target_pose.pose.orientation.z = 0.662
goal.target_pose.pose.orientation.w = 0.750

navclient.send_goal(goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
finished = navclient.wait_for_result()

if not finished:
    rospy.logerr("Action server not available!")
else:
    rospy.loginfo(navclient.get_result())
