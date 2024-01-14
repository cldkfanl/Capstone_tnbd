#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray, String



status = True
# status = False면 메인루프 정지
# status = True면 메인루프 실행

# Goal Position 리스트
goal_positions = [
    {
        'x': -0.5315938984188207,
        'y': 0.7252667771079371,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.39408683068442196,
        'qw': 0.919073212470643
    },
    {
        'x': 0.3849080244271294,
        'y': 1.7744349706434361,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': -0.4097996428677951,
        'qw': 0.9121755602434367
    },
    {
        'x': 0.8750092034281083,
        'y': 1.0277894949145323,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.9124767569988788,
        'qw': 0.4091285469590325
    },
    {
        'x': 0.08071552567121923,
        'y': 0.14939131341943943,
        'z': 0.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.9228055299838844,
        'qw': 0.38526608185922906
    }
]

# Goal Position 상태
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

# Initialpose 함수
def set_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    initpose_msg = PoseWithCovarianceStamped()
    initpose_msg.header.frame_id = "map"
    initpose_msg.pose.pose.position.x = 0.07339318096637726
    initpose_msg.pose.pose.position.y = 0.030580442398786545
    initpose_msg.pose.pose.orientation.x = 0.0
    initpose_msg.pose.pose.orientation.y = 0.0
    initpose_msg.pose.pose.orientation.z = 0.9238795336521503
    initpose_msg.pose.pose.orientation.w = 0.38268342961080154
    rospy.sleep(1)
    rospy.loginfo("Setting initial pose")
    pub.publish(initpose_msg)
    rospy.loginfo("Initial pose SET")

# Amclpose_2_Initialpose 함수
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

# Move Goal Position 함수
    def send_navigation_goal(client, x, y, z, qx, qy, qz, qw):
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

        client.send_goal(goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)
        client.wait_for_result()

# Roate 함수        
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

## 콜백함수 
    def mj_depth_callback(data):
        global status
        status= False
        global initpose_msg 
        x_values = data.data[:len(data.data)//2] # 첫 번째 절반은 x_values
        z_values = data.data[len(data.data)//2:]  # 나머지 절반은 z_values

        current_x = initpose_msg.pose.pose.position.x
        current_y = initpose_msg.pose.pose.position.y
        current_orientation = initpose_msg.pose.pose.orientation
        (current_roll, current_pitch, current_yaw) = euler_from_quaternion([current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w])

        # 좌측 또는 우측으로 x cm 이동
        x_m = - 0.012168 -0.0151599566389232

        # 앞쪽으로 z cm 이동
        y_m = z_values[0]*10 + 0.2

        # 새로운 위치 계산
        object_x = current_x + y_m * np.cos(current_yaw)
        object_y = current_y + y_m * np.sin(current_yaw)
        new_x = object_x - x_m * np.sin(current_yaw)
        new_y = object_y + x_m * np.cos(current_yaw)
        new_yaw = np.arctan2(current_y - new_y, current_x - new_x)

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


        # rospy.init_node('send_goal')

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
        status = True
        

    # def mj_type_callback(msg):
    #     global status
    #     status = False

    #     if msg == "dog" or "cat"
    #         rotate_robot(0.5, counterclockwise_180_degrees, 4.0)         
    #     status = True
               
    def control_nav_callback(msg):
        status = True


    def app_topic_callback(msg):
        global status
        status = False
        status = True


## 퍼블리셔 함수
    def set_linear():
        global linear_msg
        pub = rospy.Publisher("linear", Int32, queue_size=10)

        linear_msg = Int32()
        linear_msg.data = 1900
        linear_msg.data = 1035

        pub.publish(linear_msg)

    def set_hatch():
        global hatch_msg
        hatch_pub = rospy.Publisher("topic", Message, queue_size=10)

        work = Message()
        work.mode = 1
        work.mode = 2
        work.mode = 3
        work.mode = 4
        work.mode = 5
        work.mode = 6
        work.mode = 7

        hatch_pub.publish(work)



# 메인함수
def main():
    rospy.init_node("auto_moving_node")

    set_initial_pose()
    navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    navclient.wait_for_server()

    rate = rospy.Rate(10)  # 메인 루프의 주기를 10Hz로 설정

    # rospy.Subscriber("mj_depth", Float32MultiArray, mj_depth_callback)
    # rospy.Subscriber("mj_type", String, mj_type_callback)
    rospy.Subscriber('control_nav', String, control_nav_callback)
    rospy.Subscriber('app_topic', String, app_topic_callback)

    while not rospy.is_shutdown():
        # 콜백 함수에서 status가 True로 설정될 때까지 대기
        while not rospy.is_shutdown() and not status:
            rospy.sleep(1)  # 1초 동안 대기

        for position in goal_positions:
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




# 메인일 때 
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

