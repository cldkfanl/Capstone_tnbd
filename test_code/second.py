#!/usr/bin/env python

import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
import actionlib
import time


from actionlib_msgs.msg import GoalID
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32MultiArray, String, Int32
from hatch.msg import Message  

status = True
navclient = None 
current_goal = None  

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


# 목표위치
goal_positions = [
    {
        'x': 0.41534551978111267,
        'y': 0.3284127712249756,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.9483547323050049,
        'qw': 0.3172117616273118
    },
    {
        'x': -0.25929000973701477,
        'y': 0.8360782265663147,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.09915877959096396,
        'qw': 0.9945892062362046
    },
    {
        'x': 1.291813850402832,
        'y': 0.9074732661247253,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': -0.9884039361656453,
        'qw': 0.1518474858937709
    },
    {
        'x': 0.7882055355882147,
        'y': 0.05736984312534332,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.785510591467577,
        'qw': 0.618848212966845
    }
]

# 함수
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

def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    initpose_msg = PoseWithCovarianceStamped()
    initpose_msg.header.frame_id = "map"
    initpose_msg.pose.pose.position.x = -0.04507629945874214
    initpose_msg.pose.pose.position.y = 0.05736984312534332
    initpose_msg.pose.pose.orientation.x = 0.0
    initpose_msg.pose.pose.orientation.y = 0.0
    initpose_msg.pose.pose.orientation.z = 0.785510591467577
    initpose_msg.pose.pose.orientation.w = 0.618848212966845
    rospy.sleep(1)
    rospy.loginfo("Setting initial pose")
    pub.publish(initpose_msg)
    rospy.loginfo("Initial pose SET")

def publish_move_base_cancel():
    cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
    cancel_msg = GoalID()
    
    cancel_pub.publish(cancel_msg)

def rotate_robot(rotation_speed, rotation_direction, duration):

    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    # Set the angular (z-axis) velocity for rotation
    twist.angular.z = rotation_speed * rotation_direction
    # Get the current time
    start_time = time.time()
    # Continue publishing the Twist message for the specified duration
    while (time.time() - start_time) < duration:
        cmd_vel_pub.publish(twist)
        rospy.sleep(0.1)
    # Stop the robot by publishing zero velocity
    twist.angular.z = 0
    
    cmd_vel_pub.publish(twist)



def hatch(mode):
    hatch_pub = rospy.Publisher("topic", Message, queue_size=10) 
    work = Message()
    work.mode = mode
    hatch_pub.publish(work)
    rospy.loginfo("해치 동작 완료")


# 콜백함수
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

def mj_depth_callback(data):
    global status
    global navclient

    status = False
    publish_move_base_cancel()
    global initpose_msg 
    x_values = data.data[:len(data.data)//2] # 첫 번째 절반은 x_values
    z_values = data.data[len(data.data)//2:]  # 나머지 절반은 z_values

    current_x = -0.04507629945874214
    current_y = 0.05736984312534332
    (current_roll, current_pitch, current_yaw) = euler_from_quaternion([0, 0, 0.785510591467577, 0.618848212966845])

    # 좌측 또는 우측으로 x cm 이동
    x_m = - 0.012168 -0.0151599566389232

    # 앞쪽으로 z cm 이동
    y_m = z_values[0]*10 -0.1

    # 새로운 위치 계산
    object_x = current_x + y_m * np.cos(current_yaw)
    object_y = current_y + y_m * np.sin(current_yaw)
    new_x = object_x - x_m * np.sin(current_yaw)
    new_y = object_y + x_m * np.cos(current_yaw)
    new_yaw = np.arctan2(- current_y + new_y,  - current_x + new_x)

    new_orientation = [0, 0, np.sin(new_yaw / 2), np.cos(new_yaw / 2)]

    #새로 계산한 쿼터니언 값 출력
    new_orientation_x = new_orientation[0]
    new_orientation_y = new_orientation[1]
    new_orientation_z = new_orientation[2]
    new_orientation_w = new_orientation[3]

    print("Received x_values:", x_values)
    print("Received z_values:", z_values)
    print(new_x)
    print(new_y)

    navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    navclient.wait_for_server()

    # Example of navigation goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = new_x
    goal.target_pose.pose.position.y = new_y
    goal.target_pose.pose.position.z = 0.0
    goal.target_pose.pose.orientation.x = new_orientation_x
    goal.target_pose.pose.orientation.y = new_orientation_y
    goal.target_pose.pose.orientation.z = new_orientation_z
    goal.target_pose.pose.orientation.w = new_orientation_w 

    navclient.send_goal(goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
    finished = navclient.wait_for_result()

    linear_speed = 0.2  # 필요에 따라 속도 조정
    linear_direction = 1  # 1은 전진, -1은 후진
    duration = 5.0  # 이동 지속 시간 (초)

    # 로봇 회전
    rotate_robot(0.5, -2, 3.5)  # 180도 회전
    rospy.sleep(1.0)

    hatch(4)
    rospy.sleep(1.1)
    hatch(4)
    rospy.sleep(1.0)
    hatch(5)
    rospy.sleep(3.0)
    hatch(6)
    rospy.sleep(1.0)
    hatch(7)
    rospy.sleep(1.0)
    status = True
    
    rate = rospy.Rate(10)  

    while not rospy.is_shutdown():
        while not rospy.is_shutdown() and not status:
            rospy.sleep(1) 
        for position in goal_positions:
            if status:  
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
            rate.sleep()  

# 메인함수
def main():
    global navclient

    rospy.init_node("second_node")

    set_initial_pose()
    rospy.Subscriber('mj_depth', Float32MultiArray, mj_depth_callback)

    navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    navclient.wait_for_server()


    rospy.Subscriber('control_nav', String, control_nav_callback)


    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
