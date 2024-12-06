#!/usr/bin/env python

# Import libraries
import numpy as np
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry 
from ackermann_msgs.msg import AckermannDriveStamped



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

        self.bubble_radius = 200
        self.preprocess_conv_size = 3
        self.best_point_conv_size = 90
        self.max_lidar_dist = 10.0
        self.fast_speed = 5.1
        self.straights_speed = 5.0
        self.corners_speed = 3.0
        self.tight_steering_angle = 0.3 #0.24
        self.straights_steering_angle = 0.18 #0.16
        self.fast_steering_angle = 0.08
        self.light_steering_angle = 0.04
        self.safe_threshold = 5
        self.max_steer = 1
        self.radians_per_elem = 0
        self.prev_angle = 0.0


    def odom_callback(self, data: Odometry):
        self._current_speed = data.twist.twist.linear.x

    def scan_callback(self, data: LaserScan):
        print(self._current_speed)
        proc_ranges = self.preprocess_lidar(data.ranges)
        closest = proc_ranges.argmin()
        
        ########################################################################
        # PLANNING
        ########################################################################

        '''
        Implement planning stack here.
        '''

        # Eliminate all points inside 'bubble' (set them to zero)

        min_index = closest - self.bubble_radius
        max_index = closest + self.bubble_radius
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))

        
        ########################################################################
        # CONTROL
        ########################################################################

        '''
        Implement control stack here.
        '''

        # Vehicle control
        delta_angle = 0
        if abs(steering_angle) > self.straights_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(self.prev_angle / steering_angle)
            if delta_angle < 0.9: #brake
                self.prev_angle = steering_angle
                steering_angle *= (1.1/delta_angle)
                target_speed *= 0.8
            elif delta_angle < 1.02:
                self.prev_angle = steering_angle
                steering_angle *= 1.1
            else:
                self.prev_angle = steering_angle
                steering_angle *= 0.9
        elif abs(steering_angle) > self.straights_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(self.prev_angle / steering_angle)
            if delta_angle < 0.95: #brake
                self.prev_angle = steering_angle
                target_speed *= (delta_angle * 0.75)
                steering_angle *= 1.2
            elif delta_angle < 1.02:
                self.prev_angle = steering_angle
                steering_angle *= 1.02
            else:
                self.prev_angle = steering_angle
                steering_angle *= 0.9
        elif abs(steering_angle) > self.fast_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(self.prev_angle/ steering_angle)
            if delta_angle < 1: #brake
                self.prev_angle = steering_angle
                target_speed *= (delta_angle * 0.75)
            elif delta_angle > 2:
                self.prev_angle = steering_angle
                steering_angle *= 0.9
            else:
                self.prev_angle = steering_angle
        elif abs(steering_angle) > self.light_steering_angle:
            target_speed = self.fast_speed
            delta_angle = abs(steering_angle / self.prev_angle)
            if delta_angle < 1: #brake
                self.prev_angle = steering_angle
                target_speed *= 0.9
            else:
                self.prev_angle = steering_angle
        else:
            target_speed = self.fast_speed
            self.prev_angle = steering_angle
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        drive_msg.drive.speed = target_speed / 5 # [-1, 1]

        self._cmd_drive_pub.publish(drive_msg)
        
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), 'same') / self.preprocess_conv_size
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
        return proc_ranges
        
    def find_max_gap(self, free_space_ranges):
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        # print(slices)
        # max_len = slices[-1].stop - slices[-1].start
        # chosen_slice = slices[-1]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[::-1]:
            print(sl)
            sl_len = sl.stop - sl.start
            if sl_len > self.safe_threshold:
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop
            
    def find_best_point(self,start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.best_point_conv_size),
                                    'same') / self.best_point_conv_size
        return averaged_max_gap.argmax() + start_i     

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the lidar data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2.0
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        return steering_angle  

def main(args=None):
    rclpy.init(args=args)
    print("Control Initialized")
    control_node = Control()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
