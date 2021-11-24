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
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
import numpy as np
 






def Callback(msg):
    	print("Received pose and twist data of the car!")
 
        # Save your OpenCV2 image as a jpeg 
        time = msg.header.stamp
	 
       # cv2.imwrite('testR/'+str(time)+'.jpeg', cv2_img)

	
        
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)

        pose_pos = [msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z] 
	pose_pos = ['%.3f' % n for n in pose_pos]

	pose_ori = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
	pose_ori = ['%.3f' % n for n in pose_ori]

	twist_linear = [msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z,msg.twist.twist.linear.z]
	twist_ang = [msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z]
	twist_linear =['%.3f' % n for n in twist_linear]
	twist_ang = ['%.3f' % n for n in twist_ang]

	pose_twist_dict[str(int(str(time))/1000)]=np.array(pose_pos+pose_ori+twist_linear+twist_ang)
 
	 
	 
	f.write("\n" +str(date)+" pose_pos"+ str(pose_pos) +" pose_ori"+ str(pose_ori)+" twist_linear"+ str(twist_linear)+" pose_ang"+ str(twist_ang)+"\n")
	print(str(date))
        rospy.sleep(0.1)   #  the pubrate is 600   you only take 5HZ   0.2 = 1/5


def main():
    rospy.init_node('pose_twist_listener')




    # Define your image topic
     
    #left_sub = message_filters.Subscriber(image_topic_left, Image)
    #rospy.Subscriber(image_topic_right, Image, image_callback)
    
    rospy.Subscriber("/rr_openrover_driver/odom_encoder", Odometry, Callback);

    # Spin until ctrl + c
    rospy.spin()
    

if __name__ == '__main__':
	#stampfile = open("/home/tud-jxavier/data/imu_stamp.txt", 'a')  
	f = open("/media/tud-jxavier/SSD/data/Pose_twist/pose_twist_data.txt", 'a')   
	f.truncate(0)
	pose_twist_dict ={}
	pose_twist_path ='/media/tud-jxavier/SSD/data/Pose_twist/pose_twist_data.npz'
	main()
	np.savez(pose_twist_path,pose_twist_dict)
        #stampfile.close()
	f.close()






















 
