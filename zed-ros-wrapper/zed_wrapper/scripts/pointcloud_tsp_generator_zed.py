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
    rospy.init_node('dummy_zedpcd_tsp_generator')
 
    rospy.spin()
    

if __name__ == '__main__':
 
	main()
        

	# change to standard 16 unit int timestamp

	#[os.rename(f, ''.join(x for x in f if x.isdigit())[:16]+'.pcd') for f in os.listdir('/media/tud-jxavier/SSD/data/zed/point_cloud/') if f.endswith('.pcd')]



	
	newlist = [f[:-4]+" dummy "+"\n" for f in os.listdir('/media/tud-jxavier/SSD/data/zed/point_cloud/') if f.endswith('.pcd')]



 
	f2 = open("/media/tud-jxavier/SSD/data/zed/timestamps/timestamp_pcd_zed.txt", 'a')
	f2.truncate(0)
	f2.writelines(newlist)






















 
