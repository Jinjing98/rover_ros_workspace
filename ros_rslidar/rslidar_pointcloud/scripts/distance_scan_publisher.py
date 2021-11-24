#! /usr/bin/python
 

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan
import rospy
import time
import numpy as np
#from numpy import *
import sys






def callback_pointcloud(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data, field_names=("x", "y", "z","intensity"), skip_nans=False)
    XYdata_array = np.array(list(gen))[:,0:2] #.shape# 78750,2  #4 refer to x,y
    #num4points = XYdata_array.shape[0]
    #dis_array = np.full((XYdata_array.shape[0]), 100, dtype=float)#100 is the safe 100m
    dis = np.sqrt(np.sum(np.square(XYdata_array), axis=1)) 
    #dis = dis.tolist()
    #  now we can use final_2d to intuitivley extract the depth or given row i in the world!!!
    #print("len of dis  ",  len(dis) )
    if len(dis)==78750:
            #print("len of dis  ",  dis.shape)#, dis[337500])
    #print("idxs for this row: ", idxx4rows)
        diss4rowsMN  = dis[idxx4rows]
    #print(diss4rowsMN)
        dis4thisrow = np.nanmin(diss4rowsMN,axis=0)


    #dis4thisrow = dis[idx4thisrow]

        dis_msg = LaserScan()
        dis_msg.header.stamp = data.header.stamp#rospy.Time.now()#  




        dis_msg.header.frame_id = "rslidar"
        dis_msg.range_min = 0
        dis_msg.range_max = 60
        dis_msg.angle_increment = np.radians(120.0/625)# we know there are 625 points for each row
        dis_msg.angle_min = np.radians(-60)
        dis_msg.angle_max = np.radians(60)# from the manual, we know there are 120 view of angle
        dis_msg.ranges = np.flip(dis4thisrow,axis = 0)


        pub.publish(dis_msg)
    else:
	print("this frame is not stable! skip and not publishing scan for this frame! len of dis: ",  len(dis))






# max you can get ~10 frames/sec, since that's the rate of /rslidar_points!

    time.sleep(0.01)#0.01, try to pub as much as possible, when debug, u can set this to 1 sec  #  in non debug this should be quite small so as to real time pub!
    #print XYdata_array.shape,len(dis),dis4thisrow.shape#dis.shape
    #print dis4thisrow
    #print XYdata_array[-100:,]
 

def main():
    i = 0
    rospy.init_node('pcdlisten__dispub', anonymous=True)
    rospy.Subscriber('/rslidar_points', PointCloud2, callback_pointcloud)
    i = i+1
    
    rospy.spin()

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("usage: distance_node.py M N !  (M<N, and in 0~125)")
    else:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        


        pub = rospy.Publisher('/jj_Scan',LaserScan,queue_size = 1)


#[0,625,1250,...125*625]
#we consider from row 61 to row 66, total 6 rows.  the id is from 60*625 to(66*625-1) total 625*6 points
        stdorderlist = []
        for i in range(5):
            for j in range(125):
                stdorderlist.append(5*j+i)

	stdorderlist4oddrow = np.array(stdorderlist).reshape(-1,125)[::-1]        
	stdorderlist_2d = [stdorderlist for i in range(126)]
	stdorderlist4oddrow_2D = np.array([stdorderlist4oddrow for i in range(63)]).reshape(-1,625)
        stdorderarray_2d = np.array(stdorderlist_2d)
	#print(stdorderarray_2d.shape,stdorderlist4oddrow_2D.shape,"lllll")
	stdorderarray_2d[1::2,:] = stdorderlist4oddrow_2D

        rowarray = np.array(range(126))
        array625 = np.full((126,625),625)
        final_2d =  np.multiply(array625,rowarray[:,np.newaxis])#shape: 126,625
        final_2d = stdorderarray_2d+final_2d
	print(final_2d.shape)
        final_2d[1::2, :] = final_2d[1::2, ::-1]
#final_2d.tolist()
 #######
        #i = 0# 0-125 you can set i yourself, which refers to the ith row.
        #idx4thisrow = final_2d[i]
# also another mode: we consider from row m to row n(0-125 total 126 rows), chose the smallest dist on that colomn as the final value for each col (total 625 cols)
#but this increase cpu, you have to compute this in call back function.
#when you set M=N, it is actually like mode 1, singal row scan
#M = 60
#N = 65# WE CONDIER THE MIDDLE 6 ROWS AS EXAMPLE
        idxx4rows = final_2d[M:(N+1)]  

 
        main()
 





















 
