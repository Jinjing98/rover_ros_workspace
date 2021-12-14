# -*- coding: UTF-8 -*-
import pclpy
import sys
from pclpy import pcl
# pcd 
# instantiate a point cloud object of the specified type and read the file into the object




if len(sys.argv) < 2:
        print("Provide the PCD full path!")
else:
        data_path = sys.argv[1]
 


obj=pclpy.pcl.PointCloud.PointXYZRGBA()
pcl.io.loadPCDFile(data_path,obj)


#  
viewer=pcl.visualization.PCLVisualizer('PCD viewer')
# Set the initial angle of view, without writing viewer.setCameraPosition(0,0,-3.0,0,-1,0)
# Set the display axis, you can not write viewer.addCoordinateSystem(0.5)
viewer.addPointCloud(obj)
while(not viewer.wasStopped()):
    viewer.spinOnce(100)

