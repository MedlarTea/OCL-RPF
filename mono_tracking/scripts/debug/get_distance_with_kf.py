#! /usr/bin/env python3

import os
import rospy
import ros_numpy
import cv2
import time
import numpy as np
import shutil
# from sensor_msgs.msg import 
# from geometry_msgs import Point
# from geometry_msgs import Vector3
from mono_tracking.msg import Track
from mono_tracking.msg import TrackArray
from tensorboardX import SummaryWriter

_dir = os.path.split(os.path.realpath(__file__))[0]

class Monitor:
    def __init__(self):
        TRACKS_TOPIC = "/mono_tracking/tracks"
        self.boxesSub = rospy.Subscriber(TRACKS_TOPIC, TrackArray, self.tracksCallback)
        self.nums = 0
        self.store_dir = "/home/jing/Data/Dataset/FOLLOWING/evaluation-of-width-distance/wKF"
        # To be monitored

        rospy.loginfo("GET DISTANCE WITH KF!")
        rospy.spin()
        
    

    def tracksCallback(self, trackArrayMsg):
        for track in trackArrayMsg.tracks:
            if track.id ==1:
                continue
            np.savetxt(os.path.join(self.store_dir, "{:05d}.txt".format(self.nums)), [track.pos.x, track.pos.y])


        self.nums+=1


if __name__ == '__main__':
    rospy.init_node('DISTANCE', anonymous=True)
    monoDetector = Monitor()



