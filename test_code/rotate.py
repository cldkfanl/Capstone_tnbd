#!/usr/bin/env python

# 로봇을 회전

import rospy
from geometry_msgs.msg import Twist
import time

def rotate_robot(rotation_speed, rotation_direction, duration):
    rospy.init_node('rotate_robot_node', anonymous=True)

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

    # Shutdown the ROS node
    rospy.signal_shutdown("Rotation completed")



if __name__ == '__main__':
    try:
        # Specify rotation parameters
        rotation_speed = 0.5  # Adjust the speed as needed (positive for clockwise, negative for counterclockwise)
        rotation_direction = 1  # 1 for clockwise, -1 for counterclockwise
        duration = 5.0  # Rotation duration in seconds


        # Call the rotate_robot function with the specified parameters
        clockwise_90_degrees = 1    # 시계방향
        counterclockwise_90_degrees = -1    # 반시계방향
        clockwise_360_degrees = 4
        counterclockwise_180_degrees = -2

        ## rotate_robot(rotation_speed, rotation_direction, duration)

        # Rotate right 90 degrees
        rotate_robot(0.5, clockwise_90_degrees, 3.0)  # Rotate at 0.5 rad/s for 3 seconds

        # Pause for 3 seconds
        rospy.sleep(3.0)

        # Rotate left 90 degrees
        rotate_robot(0.5, counterclockwise_90_degrees, 3.0)  # Rotate at 0.5 rad/s for 3 seconds

        # Pause for 3 seconds
        rospy.sleep(3.0)

        # Rotate 360 degrees (clockwise)
        rotate_robot(0.5, clockwise_360_degrees, 6.0)  # Rotate at 0.5 rad/s for 6 seconds

        # Pause for 3 seconds
        rospy.sleep(3.0)

        # Rotate 180 degrees (counterclockwise)
        rotate_robot(0.5, counterclockwise_180_degrees, 4.0)  # Rotate at 0.5 rad/s for 4 seconds

    except rospy.ROSInterruptException:
        pass