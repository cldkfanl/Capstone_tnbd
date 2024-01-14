#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from hatch.msg import Message

def main():
    rospy.init_node("combined_pub")
    
    # Publishers
    linear_pub = rospy.Publisher("linear", Int32, queue_size=10)
    hatch_pub = rospy.Publisher("topic", Message, queue_size=10)
    
    rospy.loginfo("Publishing to topics linear and topic...")
    
    rate = rospy.Rate(10)  # 10 Hz
    
    msg = Int32()
    work = Message()

    while not rospy.is_shutdown():
        ch = input("Enter 'o', 'p' for linear, 'f', 'g', or 'h' for hatch: ")

        if ch.lower() in ['o', 'p']:
            if ch.lower() == 'o':
              msg.data = 1900  
            elif ch.lower() == 'p':
              msg.data = 1035  

            rospy.loginfo("Published linear message: %d", msg.data)
            linear_pub.publish(msg)
        elif ch.lower() in ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f']:
            if ch.lower() == 'q':
                work.mode = 1
                # linear 1900
            elif ch.lower() == 'w':
                work.mode = 2
                # linear 1035
            elif ch.lower() == 'e':
                work.mode = 3
                
            elif ch.lower() == 'r':
                work.mode = 4
            elif ch.lower() == 'a':
                work.mode = 5
            elif ch.lower() == 's':
                work.mode = 6
            elif ch.lower() == 'd':
                work.mode = 7
            rospy.loginfo("Published topic message: %d", work.mode)
            hatch_pub.publish(work)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
