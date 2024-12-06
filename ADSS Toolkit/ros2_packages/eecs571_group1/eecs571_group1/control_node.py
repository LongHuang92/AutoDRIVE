#!/usr/bin/env python

# Import libraries
import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry 
from ackermann_msgs.msg import AckermannDriveStamped

from .localmap_racing.localMPCC import LocalMPCC

class Control(Node):
    def __init__(self):
        super().__init__('control_node')

        lidarscan_topic = '/scan'
        odom_topic = '/odom'
        drive_topic = '/drive'
        
        self._current_speed = 0

        self._laser_scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self._vel_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self._cmd_drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        test_id = "mu60"
        self.planner = LocalMPCC(test_id, True)


    def odom_callback(self, data: Odometry):
        self._current_speed = data.twist.twist.linear.x

    def scan_callback(self, data: LaserScan):
        print(self._current_speed)
        observation = {"scan": np.array(data.ranges[:1080]),
                "vehicle_speed": self._current_speed}
        action = self.planner.plan(observation)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(action[0])*2
        drive_msg.drive.speed = float(action[1])*1

        self._cmd_drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("Control Initialized")
    control_node = Control()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
