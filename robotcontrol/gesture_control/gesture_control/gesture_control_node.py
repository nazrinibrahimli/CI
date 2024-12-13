import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("192.168.177.129", 12345))  # 确保IP地址和端口号匹配
        self.get_logger().info("Listening for commands...")

    def listen_and_publish(self):
        while rclpy.ok():
            data, _ = self.socket.recvfrom(1024)
            command = data.decode()
            self.get_logger().info(f"Received command: {command}")
            twist = Twist()

            if command == "OPEN_HAND":  # 张开手掌
                twist.linear.x = 0.0
                twist.angular.z = 0.0  # 停止
                print("停止")
            elif command == "FIST":  # 握拳
                twist.linear.x = 0.3 # 前进
                twist.angular.z = 0.0
                print("向前进")
            elif command == "THUMB_UP":  # 竖大拇指
                twist.linear.x = -0.3  # 向后退
                twist.angular.z = 0.0
                print("向后退")
            elif command == "LEFT_SWIPE":  # 张开手掌向左挥动
                twist.angular.z = 0.5 # 左转
                print("向左转")
            elif command == "RIGHT_SWIPE":  # 张开手掌向右挥动
                twist.angular.z = -0.5  # 右转
                print("向右转")

            # 发布指令到 /cmd_vel
            self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()

    try:
        node.listen_and_publish()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.socket.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
