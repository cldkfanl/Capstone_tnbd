#!/usr/bin/env python


import rospy
from std_msgs.msg import Int32

def main():
    rospy.init_node("linear_pub")
    pub = rospy.Publisher("linear", Int32, queue_size=10)
    rospy.loginfo("Publishing to topic linear...")

    rate = rospy.Rate(0.1)  
    rospy.sleep(10)  
    msg = Int32()

    while not rospy.is_shutdown():
        ch = input()
        if ch.lower() in ['o', 'p']:
            if ch.lower() == 'o':
                msg.data = 1900
            elif ch.lower() == 'p':
                msg.data = 1035

            rospy.loginfo("Published message: %d", msg.data)
            pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

