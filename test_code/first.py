#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from hatch.msg import Message  

# 이동 함수 cmd_vel
def move_robot(linear_speed, linear_direction, duration):
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    # 속도 설정
    twist.linear.x = linear_speed * linear_direction
    start_time = rospy.Time.now()

    # 주어진 지속 시간 동안 Twist 메시지를 주기적으로 게시하여 로봇을 이동시킴
    while (rospy.Time.now() - start_time).to_sec() < duration:
        cmd_vel_pub.publish(twist)
        rospy.sleep(0.1)

    twist.linear.x = 0
    cmd_vel_pub.publish(twist)
    rospy.loginfo("이동 완료")

# 회전 함수 cmd_vel
def rotate_robot(rotation_speed, rotation_direction, duration):
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    # Set the angular (z-axis) velocity for rotation
    twist.angular.z = rotation_speed * rotation_direction
    start_time = rospy.Time.now()
    
    # 주어진 지속 시간 동안 Twist 메시지를 주기적으로 게시하여 로봇을 회전시킴
    while (rospy.Time.now() - start_time).to_sec() < duration:
        cmd_vel_pub.publish(twist)
        rospy.sleep(0.1)

    # 회전이 완료되면 로봇을 정지시키고 ROS 노드를 종료
    twist.angular.z = 0
    cmd_vel_pub.publish(twist)
    rospy.loginfo("회전 완료")

# 해치 함수
def hatch(mode):
    hatch_pub = rospy.Publisher("topic", Message, queue_size=10) 
    work = Message()
    work.mode = mode
    hatch_pub.publish(work)
    rospy.loginfo("해치 동작 완료")

def linear(data):
    linear_pub = rospy.Publisher("linear", Int32, queue_size=10)
    msg = Int32()
    msg.data = data
    linear_pub.publish(msg)
    rospy.loginfo("리니어 동작 완료: %s", msg.data)

if __name__ == '__main__':
    try:
        # 노드 초기화
        rospy.init_node('robot_control_node', anonymous=True)

        # 이동 매개변수
        linear_speed = 0.15  # 필요에 따라 속도 조정
        linear_direction = 1  # 1은 전진, -1은 후진
        duration = 5.0  # 이동 지속 시간 (초)

        # 로봇을 전진
        move_robot(linear_speed, linear_direction, duration)
        rospy.sleep(0.1)  # 일시 정지 시간 추가

        # 전면해치
        hatch(1)
        rospy.sleep(1.0)

        linear(1900)            # 첫번째 데이터 씹힘
        rospy.sleep(0.1)
        linear(1100)
        hatch(1)
        rospy.sleep(1)
        linear(1900)
        rospy.sleep(5.0)

        hatch(2)
        rospy.sleep(4.0)

        linear(1035)
        rospy.sleep(5.0)
        hatch(3)
        rospy.sleep(3.0)

        # 로봇을 후진
        move_robot(-linear_speed, linear_direction, duration)
        rospy.sleep(1.0)


    except rospy.ROSInterruptException:
        pass
