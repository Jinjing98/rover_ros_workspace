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
 






def imuCallback(msg):
	#now = datetime.datetime.now().time() # time object
	#today = datetime.date.today()

# dd/mm/YY



    	print("Received an imu data!")
	print(msg.header.stamp)
	timeIMU = int(str(msg.header.stamp.secs)+str(msg.header.stamp.secs))/1000
	#print(timeIMU)

  
        

	acce = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
	acce = ['%.3f' % n for n in acce]
	vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
	vel = ['%.3f' % n for n in vel]
	oritation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
	oritation = ['%.3f' % n for n in oritation]
	 
	#f.write("\n"+ str(time)+'  '+ str(date))
	time = msg.header.stamp  

	data = np.array(acce+vel+oritation)
	dict_imu_data[str(int(str(time))/1000)] = data
     
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)
	#print(str(timestamp) + str(date))
	f2.write(str(int(str(time))/1000)+' '+ str(date)+"\n")	
	f.write(str(int(str(time))/1000)+' '+ str(date)+" "+" acce_linear"+ str(acce) +" ang_vel "+ str(vel)+" orit(q xyzw) " + str(oritation)+"\n")   
        rospy.sleep(0.01)   #  the pubrate is 600   you only take 5HZ   0.2 = 1/5


def main():
    rospy.init_node('imu_listener')




    # Define your image topic
     
    #left_sub = message_filters.Subscriber(image_topic_left, Image)
    #rospy.Subscriber(image_topic_right, Image, image_callback)
    
    rospy.Subscriber("/zed2/zed_nodelet/imu/data", Imu, imuCallback);

    # Spin until ctrl + c
    rospy.spin()
    

if __name__ == '__main__':

        f2 = open("/media/tud-jxavier/SSD/data/zed/timestamps/timestamp_IMU.txt", 'a')
        f2.truncate(0) 
	f = open("/media/tud-jxavier/SSD/data/zed/imu_data/imu_data.txt", 'a')
	f.truncate(0) 
	dict_imu_data = {}  
	main()
	imu_npz_path = "/media/tud-jxavier/SSD/data/zed/imu_data/imu_data.npz"
	np.savez(imu_npz_path,**dict_imu_data)


        #stampfile.close()
	f.close()
	f2.close()





















 
