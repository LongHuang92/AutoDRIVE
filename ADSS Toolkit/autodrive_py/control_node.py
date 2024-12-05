#!/usr/bin/env python

# Import libraries
import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from localmap_racing.localMPCC import LocalMPCC

class Control(Node):
    def __init__(self):
        super().__init__('control_node')

        lidarscan_topic = '/scan'
        odom_topic = '/odom'
        drive_topic = '/drive'

        self._laser_scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self._vel_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self._cmd_drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        test_id = "mu60"
        self.planner = LocalMPCC(test_id, True)


    def odom_callback(self, data: Odometry):
        self._current_speed = data.twist.twist.linear.x

    def scan_callback(self, data: LaserScan):
        observation = {"scan": data[:1080],
                "vehicle_speed": self._current_speed}

        action = self.planner.plan(observation)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = action_1[0]*2
        drive_msg.drive.speed = action_1[1]*0.02

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