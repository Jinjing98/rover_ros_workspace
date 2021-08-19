#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image,Imu,PointCloud2
# ROS Image message -> OpenCV2 image converter
# from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
import numpy as np
 






def pcCallback(msg):
    	print("Received an pointcloud data!")
 
        # Save your OpenCV2 image as a jpeg 
        time = msg.header.stamp
	height = msg.height
	width = msg.width
	row_step = msg.row_step
	fields = msg.fields
	 
       # cv2.imwrite('testR/'+str(time)+'.jpeg', cv2_img)
        
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)

	
	 

	#acce = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
	#vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
	#vel = ['%.3f' % n for n in vel]
	#oritation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
	#oritation = ['%.3f' % n for n in oritation]
	 
	#f.write("\n"+ str(time)+'  '+ str(date))
	#f.write("\n" +str(date)+ Data +" acceXYZ"+ str(acce) +" angVelXYZ "+ str(vel)+" oritXYZW " + str(oritation))
	# 
	f.write("\n" +str(date)+"\n" +str(height)+"\n" +str(width)+"\n" +str(row_step)+"\n"+str(fields) )
	 
	rospy.sleep(0.2)   #  the pubrate is 600   you only take 5HZ   0.2 = 1/5


def main():
    rospy.init_node('pointcloud_listener')




    # Define your image topic
     
    #left_sub = message_filters.Subscriber(image_topic_left, Image)
    #rospy.Subscriber(image_topic_right, Image, image_callback)
    
    rospy.Subscriber("/rslidar_points", PointCloud2, pcCallback);

    # Spin until ctrl + c
    rospy.spin()
    

if __name__ == '__main__':
	#stampfile = open("/home/tud-jxavier/data/imu_stamp.txt", 'a')  
	f = open("/media/tud-jxavier/SSD/data/rslidar/lidarData_nondatapart.txt", 'a')  
        f.truncate(0) 
	main()
        #stampfile.close()
	f.close()






















 
