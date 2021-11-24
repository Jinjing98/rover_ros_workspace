#! /usr/bin/python

import rospy
# ROS Image message
from sensor_msgs.msg import Image,Imu
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
import numpy as np
import datetime,time
from datetime import timedelta
import rospy
import time
import argparse
import sys
import os
 
import glob
 
 

def main():
    rospy.init_node('dummy_zed_pcd_cleaner')
 
    rospy.spin()
    

if __name__ == '__main__':
 


	filesR = glob.glob('/media/tud-jxavier/SSD/data/zed/point_cloud/*')
	for i in filesR:
		os.remove(i)
 
	main()
        





















 
