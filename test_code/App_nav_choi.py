#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String  
from std_msgs.msg import Int32 
from hatch.msg import Message 
from actionlib_msgs.msg import GoalID

status = True
navclient = None  # navclient를 전역 변수로 선언
current_goal = None  # 현재의 네비게이션 목표를 저장하기 위한 변수

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
        'x': 0.3,
        'y': -0.15,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': -0.2392205871446352,
        'qw': 0.9709652468992781
    },
]

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
    pass 

# Function to send navigation goal
def send_navigation_goal(client, x, y, z, qx, qy, qz, qw):
    global current_goal
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

    current_goal = goal  # 현재의 목표를 저장
    client.send_goal(goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
    client.wait_for_result()

# Function to set initial pose
def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    initpose_msg = PoseWithCovarianceStamped()
    initpose_msg.header.frame_id = "map"
    initpose_msg.pose.pose.position.x = -0.009999923408031464
    initpose_msg.pose.pose.position.y = -0.09999989718198776
    initpose_msg.pose.pose.orientation.x = 0.0
    initpose_msg.pose.pose.orientation.y = 0.0
    initpose_msg.pose.pose.orientation.z = -0.2392205871446352
    initpose_msg.pose.pose.orientation.w = 0.9709652468992781
    rospy.sleep(1)
    rospy.loginfo("Setting initial pose")
    pub.publish(initpose_msg)
    rospy.loginfo("Initial pose SET")

# 여기 보세요
def publish_move_base_cancel():
    cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
    cancel_msg = GoalID()
    
    cancel_pub.publish(cancel_msg)

def control_nav_callback(msg):
    global status
    if msg.data == "start":
        status = True

    elif msg.data == "end":
        status = False
        publish_move_base_cancel()
        # "end" 메시지를 받으면 move_base/cancel 메시지를 발행

    else:
        rospy.logwarn("Unknown control_nav message: %s", msg.data)

def main():
    global navclient

    rospy.init_node('App_nav', anonymous=True)

    set_initial_pose()
    navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    navclient.wait_for_server()

    rate = rospy.Rate(10)  # 메인 루프의 주기를 10Hz로 설정

    rospy.Subscriber('control_nav', String, control_nav_callback)

    while not rospy.is_shutdown():
        # 콜백 함수에서 status가 True로 설정될 때까지 대기
        while not rospy.is_shutdown() and not status:
            rospy.sleep(1)  # 1초 동안 대기

        for position in goal_positions:
            if status:  # status가 True일 때만 목표를 보냅니다.
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
            rate.sleep()  # 10Hz 주기로 메인 루프를 실행

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass