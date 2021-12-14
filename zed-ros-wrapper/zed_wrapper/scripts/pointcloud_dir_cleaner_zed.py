#! /usr/bin/python

 
import sys
import os
 
import glob
 
 

def main():
    rospy.init_node('dummy_zed_pcd_cleaner')
 
   # rospy.spin()
    

if __name__ == '__main__':
 


	filesR = glob.glob('/media/tud-jxavier/SSD/data/zed/point_cloud/*')
	for i in filesR:
		os.remove(i)
 
	#main()
        





















 
