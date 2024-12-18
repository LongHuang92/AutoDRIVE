#!/usr/bin/env python

# Import libraries
import socketio
import eventlet
from flask import Flask
import autodrive
import numpy as np
import math
import matplotlib.pyplot as plt

# added by Long 11/2//2024
from localmap_racing.localMPCC import LocalMPCC
test_id = "mu60"
planner = LocalMPCC(test_id, True)

################################################################################

# Initialize vehicle(s)
f1tenth_1 = autodrive.F1TENTH()
f1tenth_1.id = 'V1'

# f1tenth_2 = autodrive.F1TENTH()
# f1tenth_2.id = 'V2'

# Initialize the server
sio = socketio.Server()

# Flask (web) app
app = Flask(__name__) # '__main__'

# Registering "connect" event handler for the server
@sio.on('connect')
def connect(sid, environ):
    print('Connected!')

# position initialization
# last_position = None

# Registering "Bridge" event handler for the server
@sio.on('Bridge')
def bridge(sid, data):
    if data:
        ########################################################################
        # PERCEPTION
        ########################################################################

        # Vehicle data
        f1tenth_1.parse_data(data, verbose=True)
        # f1tenth_2.parse_data(data, verbose=True)

        '''
        Implement peception stack here.
        '''
        scan_1 = f1tenth_1.lidar_range_array
        # scan_2 = f1tenth_2.lidar_range_array

        scan_adjusted_1 = scan_1.copy()

        global last_scan_1
        if last_scan_1 is not None:
            scan_diff_1 = scan_1 - last_scan_1

            index_diff = np.where(scan_diff_1[450:650] > 1) # find points whose relative speed > 1
            index_scan = np.where(scan_1[450:650] < 2) # find points whose relative distance < 2
            index = np.intersect1d(index_diff[0], index_scan[0])
            print(index_diff[0])
            print(index_scan[0])
            print(index)

            # if index_scan[0].size > 20:
            #     scan_adjusted_1[600:1081] = 0.5
            #     print(scan_adjusted_1)

            # if index.size > 0:
            #     scan_adjusted_1[450+index] = 6 

            fig, axs = plt.subplots(4, 1, num=3, clear=True, figsize=(8, 10))
            axs[0].plot(last_scan_1)
            axs[0].set_ylim([0,10])
            axs[0].set_ylabel('Last Scan')
            axs[1].plot(scan_1)
            axs[1].set_ylim([0,10])
            axs[1].set_ylabel('Current Scan')
            axs[2].plot(scan_diff_1)
            axs[2].set_ylim([-5,10])
            axs[2].set_ylabel('Scan Difference')
            axs[3].plot(scan_adjusted_1)
            axs[3].set_ylim([-5,10])
            axs[3].set_ylabel('Adjusted Scan')
            plt.pause(0.001)

        last_scan_1 = scan_1

        # Adjust the lidar scan according to the measured obstacle speed
        # obstacle_index = moving_object_detection(camera_view) # there could be multiple moving objects 
        # obstacle_speed = track_object_speed(scan,last_scan,obstacle_index)
        # speed_threshold = 0
        # scan_adjusted = scan
        # if obstacle_speed > speed_threshold:
        #     scan_adjusted[obstacle_index] = float('inf')

        # curr_position = np.array(list(map(float, data['V1 Position'].split(" "))))
        # global last_position
        # if last_position is None:
        #     speed = 0
        # else:
        #     speed = math.sqrt((curr_position[0] - last_position[0])**2 + (curr_position[1] - last_position[1])**2)
        # last_position = curr_position
        # print('*****************')
        # print(speed)

        observation_1 = {"scan": scan_adjusted_1[:1080],
                "vehicle_speed": float(data['V1 Speed'])}
        # observation_2 = {"scan": scan_2[:1080],
        #         "vehicle_speed": float(data['V2 Throttle'])}

        ########################################################################
        # PLANNING
        ########################################################################

        '''
        Implement planning stack here.
        '''
        action_1 = planner.plan(observation_1)

        # action_2 = planner.plan(observation_2)

        ########################################################################
        # CONTROL
        ########################################################################

        '''
        Implement control stack here.
        '''

        # Vehicle control
        f1tenth_1.throttle_command = action_1[1]*0.02#0.03 # [-1, 1]
        f1tenth_1.steering_command = action_1[0]*2 # [-1, 1]

        # f1tenth_2.throttle_command = action_2[1]*0.01 # [-1, 1]
        # f1tenth_2.steering_command = action_2[0]*2 # [-1, 1]

        ########################################################################

        json_msg_1 = f1tenth_1.generate_commands(verbose=True) # Generate vehicle 1 message
        # json_msg_2 = f1tenth_2.generate_commands(verbose=True) # Generate vehicle 2 message

        try:
            sio.emit('Bridge', data= json_msg_1)
        except Exception as exception_instance:
            print(exception_instance)

        # try:
        #     sio.emit('Bridge', data= json_msg_2)
        # except Exception as exception_instance:
        #     print(exception_instance)

################################################################################

if __name__ == '__main__':
    last_scan_1 = None
    app = socketio.Middleware(sio, app) # Wrap flask application with socketio's middleware
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app) # Deploy as an eventlet WSGI server