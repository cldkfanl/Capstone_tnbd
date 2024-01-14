#! /usr/bin/env python


# amcl_pose토픽을 받았을때, initialpose값을 amcl값으로 퍼블리시
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

initpose_msg = PoseWithCovarianceStamped()

def amcl_pose_callback(msg):
    global initpose_msg

    # 새 변수 만들어서 amcl_pose값 당할
    position_x = msg.pose.pose.position.x
    position_y = msg.pose.pose.position.y
    orientation_x = msg.pose.pose.orientation.x
    orientation_y = msg.pose.pose.orientation.y
    orientation_z = msg.pose.pose.orientation.z
    orientation_w = msg.pose.pose.orientation.w

    # initialpose <- amcl_pose
    initpose_msg.header.frame_id = "map"
    initpose_msg.pose.pose.position.x = position_x
    initpose_msg.pose.pose.position.y = position_y
    initpose_msg.pose.pose.orientation.x = orientation_x
    initpose_msg.pose.pose.orientation.y = orientation_y
    initpose_msg.pose.pose.orientation.z = orientation_z
    initpose_msg.pose.pose.orientation.w = orientation_w
    initialpose_pub.publish(msg)

rospy.init_node("amcl_pose_to_initialpose")
initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)

rospy.spin()