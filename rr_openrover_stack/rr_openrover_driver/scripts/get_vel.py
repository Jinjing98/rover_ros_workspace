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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
import numpy as np
 






def Callback(msg):
    	print("Received twist data of the car!")
 
        # Save your OpenCV2 image as a jpeg 
        #time = msg.header.stamp

        time = rospy.get_rostime()
	 
       # cv2.imwrite('testR/'+str(time)+'.jpeg', cv2_img)

	
        
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)
 

	twist_linear = [msg.linear.x,msg.linear.y,msg.linear.z]
	twist_ang = [msg.angular.x,msg.angular.y,msg.angular.z]
	twist_linear =['%.3f' % n for n in twist_linear]
	twist_ang = ['%.3f' % n for n in twist_ang]

	pose_twist_dict[str(int(str(time))/1000)]=np.array(twist_linear+twist_ang)
 
	 
	#f.write("\n"+ str(time)+'  '+ str(date))
	f.write("\n" +str(date)+" "+str(int(str(time))/1000)+" twist_linear"+ str(twist_linear)+" twist_ang"+ str(twist_ang)+"\n")
	print(str(date))
        rospy.sleep(0.0000001)   #  the pubrate is 600   you only take 5HZ   0.2 = 1/5


def main():
    rospy.init_node('/vel_collection_listener')
 
    rospy.Subscriber("/cmd_vel/managed", Twist, Callback)

    # Spin until ctrl + c
    rospy.spin()
    

if __name__ == '__main__':
	rospy.init_node('vel_collection_listener')
	rospy.Subscriber("/cmd_vel/managed", Twist, Callback)

	time_file = rospy.get_rostime()  
	f = open("/media/tud-jxavier/SSD/data/Pose_twist/vel_collection.txt", 'a')   
	f.truncate(0)
	pose_twist_dict ={}	
	pose_twist_path ='/media/tud-jxavier/SSD/data/Pose_twist/vel_collection'+str(int(str(time_file))/1000)+'.npz'

	rospy.spin()
	
	np.savez(pose_twist_path,**pose_twist_dict)
        #stampfile.close()
	f.close()






















 
