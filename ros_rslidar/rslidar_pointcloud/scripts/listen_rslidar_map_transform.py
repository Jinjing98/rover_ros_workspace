#! /usr/bin/python
 
 
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan
import datetime,time
from datetime import timedelta
import rospy
import time
import numpy as np
import sys

import roslib
import rospy
import math
import tf
import geometry_msgs.msg


def callback_pose(data):
    	assert isinstance(data, PointCloud2)


    	print("Received a pose data!")
    	#listener = tf.TransformListener()
	(trans,rot) = listener.lookupTransform('map', 'rslidar', rospy.Time(0))

        # Save your OpenCV2 image as a jpeg 
	time = data.header.stamp
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)
	f.write(str(int(str(time))/1000)+" "+str(trans[0])+" "+ str(trans[1])+" "+str(trans[2])+" "+ str(rot[0])+" "+ str(rot[1]) +" "+ str(rot[2]) +" "+ str(rot[3])  +"\n")
	f2.write(str(int(str(time))/1000)+" "+str(date)+"\n")   #  /3
	 
	rospy.sleep(0.0002) # change this will adjust the freq. smaller will get more !
 
rospy.init_node('poses_listener', anonymous=True)
listener = tf.TransformListener()
def main():



    rospy.Subscriber('/rslidar_points',PointCloud2, callback_pose)
    rospy.spin()

if __name__ == "__main__":
    
    f = open("/media/tud-jxavier/SSD/data/rslidar/rslidar_map_poses.txt", 'a')
    f.truncate(0)   # this is cleanup operations
    f2 = open("/media/tud-jxavier/SSD/data/rslidar/timestamps/timestamp_poses.txt", 'a')
    f2.truncate(0)   # this is cleanup operations    
    main()
        #stampfile.close()
    f.close()
    f2.close()





















 
