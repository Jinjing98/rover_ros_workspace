#! /usr/bin/python2.7
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
from zed_interfaces.msg import Objects
import os
import glob


 


# Instantiate CvBridge
bridge = CvBridge()

def callback(imageL,imageR ):
    print("Received an sys pair of images  ")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img_L = bridge.imgmsg_to_cv2(imageL, "bgr8")
        cv2_img_R = bridge.imgmsg_to_cv2(imageR, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        timeL = imageL.header.stamp
        timeR = imageR.header.stamp
       # print('before')
        cv2.imwrite('/media/tud-jxavier/SSD/data/zed/imgLeft/'+str(int(str(timeL))/1000)+'.jpeg', cv2_img_L)
	cv2.imwrite('/media/tud-jxavier/SSD/data/zed/imgRight/'+str(int(str(timeR))/1000)+'.jpeg', cv2_img_R)
	#print(timeL==timeR)
	#time_stamp = 1605151831
	timestamp = timeL.to_time()	
	date = datetime.datetime.fromtimestamp(timestamp)
       # print(str(timestamp)+ str(date))	 
	f.write( str(int(str(timeL))/1000)+' '+ str(date)+"\n")
	
 

    
	





	
        rospy.sleep( 0.05)#max grabing is 30 or so 


def main():
    rospy.init_node('image_listener')
  
    # Define your image topic
    #image_topic = "/camera/rgb/image_raw"
    image_topic_left ="/zed2/zed_nodelet/left/image_rect_color"#"/zed2/zed_nodelet/stereo_raw/image_raw_color" # "/zed2/zed_nodelet/left/image_rect_color"
    image_topic_right = "/zed2/zed_nodelet/right/image_rect_color"
    #obj_topic = 
 


    left_sub = message_filters.Subscriber(image_topic_left, Image)
    right_sub = message_filters.Subscriber(image_topic_right, Image)
  #  obj_det_sub= message_filters.Subscriber(obj_det_topic,Objects);

 


    ts = message_filters.TimeSynchronizer([left_sub, right_sub ], 100)
    ts.registerCallback(callback)
    rospy.spin()

    





    # Set up your subscriber and define its callback
   # rospy.Subscriber(image_topic_left, Image, image_callback)
    # Spin until ctrl + c
    #rospy.spin()

if __name__ == '__main__':

    filesR = glob.glob('/media/tud-jxavier/SSD/data/zed/imgRight/*')
    filesL = glob.glob('/media/tud-jxavier/SSD/data/zed/imgLeft/*')
    for i in filesR:
        os.remove(i)
    for i in filesL:
        os.remove(i)
    f = open("/media/tud-jxavier/SSD/data/zed/timestamps/timestamp_IMG.txt", 'a')
    f.truncate(0)
 
    main()
    f.close()
 
    
