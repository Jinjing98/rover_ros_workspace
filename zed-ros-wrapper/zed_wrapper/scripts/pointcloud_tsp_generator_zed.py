#! /usr/bin/python

 
import sys
import os
 
import glob
 
 

def main():
    rospy.init_node('dummy_zedpcd_tsp_generator')
 
   # rospy.spin()
    

if __name__ == '__main__':
 
	#main()
        

	# change to standard 16 unit int timestamp

	#[os.rename(f, ''.join(x for x in f if x.isdigit())[:16]+'.pcd') for f in os.listdir('/media/tud-jxavier/SSD/data/zed/point_cloud/') if f.endswith('.pcd')]



	
	newlist = [f[:-4]+" dummy "+"\n" for f in os.listdir('/media/tud-jxavier/SSD/data/zed/point_cloud/') if f.endswith('.pcd')]



 
	f2 = open("/media/tud-jxavier/SSD/data/zed/timestamps/timestamp_pcd_zed.txt", 'a')
	f2.truncate(0)
	f2.writelines(newlist)






















 
