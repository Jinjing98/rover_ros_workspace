#! /usr/bin/python

 
import sys
import os
 
import glob
 
 

def main():
    rospy.init_node('dummy_rslidarpcd_tsp_generator')
 
  #  rospy.spin()
    

if __name__ == '__main__':
 
	#main()




	newlist = [f[:-4]+" dummy "+"\n" for f in os.listdir('/media/tud-jxavier/SSD/data/rslidar/Pointcloud/') if f.endswith('.pcd')]



 
	f2 = open("/media/tud-jxavier/SSD/data/rslidar/timestamps/timestamp_pcd_rslidar.txt", 'a')
	f2.truncate(0)
	f2.writelines(newlist)
        

        





















 
