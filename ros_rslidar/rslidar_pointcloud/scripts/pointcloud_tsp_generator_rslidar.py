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
    rospy.init_node('dummy_rslidarpcd_tsp_generator')
 
    rospy.spin()
    

if __name__ == '__main__':
 
	main()




	newlist = [f[:-4]+" dummy "+"\n" for f in os.listdir('/media/tud-jxavier/SSD/data/rslidar/Pointcloud/') if f.endswith('.pcd')]



 
	f2 = open("/media/tud-jxavier/SSD/data/rslidar/timestamps/timestamp_pcd_rslidar.txt", 'a')
	f2.truncate(0)
	f2.writelines(newlist)
        

        





















 
