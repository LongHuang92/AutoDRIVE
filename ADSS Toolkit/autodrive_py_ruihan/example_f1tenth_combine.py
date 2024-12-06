#!/usr/bin/env python

# Import libraries
import socketio
import eventlet
from flask import Flask
import autodrive
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
# Global Variables

bubble_radius = 200
preprocess_conv_size = 3
best_point_conv_size = 90
max_lidar_dist = 10.0
fast_speed = 2.0 #5.1
straights_speed = 2.0#5.0
corners_speed = 1.0#3.0
tight_steering_angle = 0.3 
straights_steering_angle = 0.18 
fast_steering_angle = 0.08
light_steering_angle = 0.04
safe_threshold = 5
max_steer = 1
radians_per_elem = 0
prev_angle = 0.0

distance_old = 1
timeread_old = 1
################################################################################

# Initialize vehicle(s)
f1tenth_1 = autodrive.F1TENTH()
f1tenth_1.id = 'V1'

# Initialize the server
sio = socketio.Server()

# Flask (web) app
app = Flask(__name__) # '__main__'

# Registering "connect" event handler for the server
@sio.on('connect')
def connect(sid, environ):
    print('Connected!')

# Registering "Bridge" event handler for the server
@sio.on('Bridge')
def bridge(sid, data):
    if data:
        
        ########################################################################
        # PERCEPTION
        ########################################################################

        # Vehicle data
        f1tenth_1.parse_data(data, verbose=False)
        '''
        Implement peception stack here.
        '''
        # Find closest point to LiDAR
        proc_ranges = preprocess_lidar(f1tenth_1.lidar_range_array)
        closest = proc_ranges.argmin()

        plot_lidar_readings(f1tenth_1.lidar_range_array, 1, 350)
        # Camera object detection
        y, x, _ = f1tenth_1.front_camera_image.shape
        for detection in f1tenth_1.detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = f1tenth_1.model.names[int(cls)]
            left, right = obstacle_angle(x1, y1, x2, y2, x, y)
            print(f'Image size {x} * {y}, Detected {class_name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}], relative position {180-left} to {180-right}')
            if (class_name == "car" or class_name == "truck" or class_name == "train" or class_name == "bus" or class_name == "boat"):
                # TODO: LIDAR

                plot_lidar_readings(f1tenth_1.lidar_range_array, 1, 350)
                read_range = get_car_read(f1tenth_1.lidar_range_array, 170, 190)
                obj_speed = find_speed(read_range)
                print("object speed: ", obj_speed)
            
            break


        ########################################################################
        # PLANNING
        ########################################################################

        '''
        Implement planning stack here.
        '''

        # Eliminate all points inside 'bubble' (set them to zero)

        min_index = closest - bubble_radius
        max_index = closest + bubble_radius
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = get_angle(best, len(proc_ranges))
        global prev_angle

        
        ########################################################################
        # CONTROL
        ########################################################################

        '''
        Implement control stack here.
        '''

        # Vehicle control

        # TODO: change trajectory planning
        delta_angle = 0
        if abs(steering_angle) > straights_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(prev_angle / steering_angle)
            if delta_angle < 0.9: #brake
                prev_angle = steering_angle
                steering_angle *= (1.1/delta_angle)
                target_speed *= 0.8
            elif delta_angle < 1.02:
                prev_angle = steering_angle
                steering_angle *= 1.1
            else:
                prev_angle = steering_angle
                steering_angle *= 0.9
        elif abs(steering_angle) > straights_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(prev_angle / steering_angle)
            if delta_angle < 0.95: #brake
                prev_angle = steering_angle
                target_speed *= (delta_angle * 0.75)
                steering_angle *= 1.2
            elif delta_angle < 1.02:
                prev_angle = steering_angle
                steering_angle *= 1.02
            else:
                prev_angle = steering_angle
                steering_angle *= 0.9
        elif abs(steering_angle) > fast_steering_angle:
            target_speed = 1.5 / (np.sqrt(abs(steering_angle)))
            delta_angle = abs(prev_angle/ steering_angle)
            if delta_angle < 1: #brake
                prev_angle = steering_angle
                target_speed *= (delta_angle * 0.75)
            elif delta_angle > 2:
                prev_angle = steering_angle
                steering_angle *= 0.9
            else:
                prev_angle = steering_angle
        elif abs(steering_angle) > light_steering_angle:
            target_speed = fast_speed
            delta_angle = abs(steering_angle / prev_angle)
            if delta_angle < 1: #brake
                prev_angle = steering_angle
                target_speed *= 0.9
            else:
                prev_angle = steering_angle
        else:
            target_speed = fast_speed
            prev_angle = steering_angle
        f1tenth_1.throttle_command = target_speed / 20 #5 # [-1, 1]
        f1tenth_1.steering_command = np.clip(steering_angle, -max_steer, max_steer) # [-1, 1]

        ########################################################################

        json_msg = f1tenth_1.generate_commands(verbose=True) # Generate vehicle 1 message

        try:
            sio.emit('Bridge', data=json_msg)
        except Exception as exception_instance:
            print(exception_instance)

def preprocess_lidar(ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        global radians_per_elem
        radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(preprocess_conv_size), 'same') / preprocess_conv_size
        proc_ranges = np.clip(proc_ranges, 0, max_lidar_dist)
        return proc_ranges

def find_max_gap(free_space_ranges):
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
        sl_len = sl.stop - sl.start
        if sl_len > safe_threshold:
            chosen_slice = sl
            return chosen_slice.start, chosen_slice.stop
            
def find_best_point(start_i, end_i, ranges):
    """Start_i & end_i are start and end indices of max-gap range, respectively
    Return index of best point in ranges
    Naive: Choose the furthest point within ranges and go there
    """
    # do a sliding window average over the data in the max gap, this will
    # help the car to avoid hitting corners
    averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(best_point_conv_size),
                                    'same') / best_point_conv_size
    return averaged_max_gap.argmax() + start_i     

def get_angle(range_index, range_len):
    """ Get the angle of a particular element in the lidar data and transform it into an appropriate steering angle
    """
    global radians_per_elem
    lidar_angle = (range_index - (range_len / 2)) * radians_per_elem
    steering_angle = lidar_angle / 2.0
    steering_angle = np.clip(steering_angle, -max_steer, max_steer)
    return steering_angle   

def obstacle_angle(x_min, y_min, x_max, y_max, x=640,y=360):
    x_center = x/2
    left = 0
    right = 0
    if (x_min - x_center != 0):
        left = y_min / (x_min - x_center)
    if (x_min - x_center != 0):
        right = y_min / (x_max - x_center)
    return (obstacle_angle_helper(left)+15), obstacle_angle_helper(right)

def obstacle_angle_helper(a):
    if (a < -2):
        return 180
    elif (a > 2):
        return 165
    elif (a < -0.93):
        return 195
    elif (a > 0.93):
        return 150
    elif (a < -0.54):
        return 210
    elif (a > 0.54):
        return 135
    elif (a < -0.31):
        return 225
    elif (a > 0.31):
        return 120   
    elif (a <= 0):
        return 255
    elif (a > 0):
        return 90   



def get_car_read(readings, degreeStart, degreeEnd):
    indexStart = int((degreeStart / 360) * 1081)
    indexEnd = int((degreeEnd / 360) * 1081)

    sliced_array = readings[indexStart:indexEnd]

    sliced_array = sliced_array.tolist()

    return sliced_array

def find_speed(distance_new):
    global distance_old
    global timeread_old


    median_distance = statistics.median(distance_new)
    timeread_new = time.time()
    
    i = 1
    sum = 1
    for item in distance_new:
        if abs(item - median_distance) > 1:
            continue
        else:
            i += 1
            sum += item
    
    accurate_distance = sum / i
    print(timeread_new, " ", timeread_old)
    car_speed = (accurate_distance - distance_old) / (timeread_new - timeread_old)

    distance_old = accurate_distance
    timeread_old = timeread_new

    return car_speed

def plot_lidar_readings(readings, degreeStart, degreeEnd):
    # print("this is the plot function")
    # np.set_printoptions(threshold=np.inf)
    
    angles = np.linspace(0, 2 * np.pi, len(readings))

    x = readings * np.cos(angles)
    y = readings * np.sin(angles)

    print("lidar: ", readings.tolist())

    indexStart = int((degreeStart / 360) * 1081)
    indexEnd = int((degreeEnd / 360) * 1081)

    plt.figure(num=1, figsize=(8, 8))
    plt.scatter(x[:indexStart], y[:indexStart], s=10, c='blue')
    plt.scatter(x[indexStart:indexEnd], y[indexStart:indexEnd], s=10, c='red')
    plt.scatter(x[indexEnd:], y[indexEnd:], s=1, c='blue')
    plt.title('2D Reconstruction of LiDAR Readings')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(["This is my legend"], fontsize="x-large")
    plt.show()
    plt.pause(0.001)

################################################################################

if __name__ == '__main__':
    app = socketio.Middleware(sio, app) # Wrap flask application with socketio's middleware
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app) # Deploy as an eventlet WSGI server
