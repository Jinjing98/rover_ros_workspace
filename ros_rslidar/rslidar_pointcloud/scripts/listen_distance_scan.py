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


def callback_scan(data):
    	assert isinstance(data, LaserScan)


    	print("Received a laser scan data!")
 
        # Save your OpenCV2 image as a jpeg 
        time = data.header.stamp
	distance_list = data.ranges 
	timestamp = time.to_time()	  # trans unit from nsec to sec
	date = datetime.datetime.fromtimestamp(timestamp)
	f.write("\n" +str(date)+"\n"+str(distance_list) +"\n")
	f2.write(str(int(str(time))/1000)+" "+str(date)+"\n")   #  /3
	 
	rospy.sleep(0.0002) # change this will adjust the freq. smaller will get more !
 

def main():
    rospy.init_node('scan_listener', anonymous=True)
    rospy.Subscriber('/jj_Scan', LaserScan, callback_scan)
    rospy.spin()

if __name__ == "__main__":
    f = open("/media/tud-jxavier/SSD/data/rslidar/laser_scan/scan_data.txt", 'a')
    f.truncate(0)   # this is cleanup operations
    f2 = open("/media/tud-jxavier/SSD/data/rslidar/timestamps/timestamp_scan.txt", 'a')
    f2.truncate(0)   # this is cleanup operations    
    main()
        #stampfile.close()
    f.close()
    f2.close()





















 
