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
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
from zed_interfaces.msg import Objects
import numpy as np
import glob
import os
import pickle


# Instantiate CvBridge
bridge = CvBridge()

def callback(msg):
        print("detected "+str(len(msg.objects)) +" objs in the current frame!")

	if len(msg.objects) == 0:
		return
 
        # Save your OpenCV2 image as a jpeg 
        time = msg.objects[0].header.stamp
	timestamp = time.to_time()	
	date = datetime.datetime.fromtimestamp(timestamp)
 
 








	f.write(str(int(str(time))/1000)+' '+ str(date)+"\n")# only time stamp for alighment
	
 
 
	
	#more directly see the obj info
	f2.write(str(int(str(time))/1000)+' '+ str(date)+' totalNUM_objects: '+str(len(msg.objects))+" " )
	num4objs = len(msg.objects)
#since different frames may have dofferent num of obj, we don't have fix array size for the info, we prefer list then
 

# below 4 regarding as dictionary and saved as .npy

	array_2D_vis = np.zeros((num4objs,4))
	list_vis_label = []
	list_full_info = []
	array_3D_vis = np.zeros((num4objs,6))
	array_3D_full = np.zeros((num4objs,8,3))
 # writing to an addtional txt for direct checking
	dict_everything4frame = {} 
	
	
	
    	for i in range(num4objs):
    
       	 	if(msg.objects[i].label_id == -1):
           	    continue;
 
		P1 =msg.objects[i].bbox_2d[0].x #[msg.objects[i].bbox_2d[0].x,msg.objects[i].bbox_2d[0].y]#,msg.objects[i].bbox_2d[0].z]
		P2 =msg.objects[i].bbox_2d[0].y#[msg.objects[i].bbox_2d[1].x,msg.objects[i].bbox_2d[1].y]#,msg.objects[i].bbox_2d[1].z]
		P3 =msg.objects[i].bbox_2d[2].x#[msg.objects[i].bbox_2d[2].x,msg.objects[i].bbox_2d[2].y]#msg.objects[i].bbox_2d[2].z]
		P4 =msg.objects[i].bbox_2d[2].y#[msg.objects[i].bbox_2d[3].x,msg.objects[i].bbox_2d[3].y]#msg.objects[i].bbox_2d[3].z]

 
		Q1 = [msg.objects[i].bbox_3d[0].x,msg.objects[i].bbox_3d[0].y,msg.objects[i].bbox_3d[0].z]
		Q2 = [msg.objects[i].bbox_3d[1].x,msg.objects[i].bbox_3d[1].y,msg.objects[i].bbox_3d[1].z]
		Q3 = [msg.objects[i].bbox_3d[2].x,msg.objects[i].bbox_3d[2].y,msg.objects[i].bbox_3d[2].z]
		Q4 = [msg.objects[i].bbox_3d[3].x,msg.objects[i].bbox_3d[3].y,msg.objects[i].bbox_3d[3].z]
		Q5 = [msg.objects[i].bbox_3d[4].x,msg.objects[i].bbox_3d[4].y,msg.objects[i].bbox_3d[4].z]
		Q6 = [msg.objects[i].bbox_3d[5].x,msg.objects[i].bbox_3d[5].y,msg.objects[i].bbox_3d[5].z]
		Q7 = [msg.objects[i].bbox_3d[6].x,msg.objects[i].bbox_3d[6].y,msg.objects[i].bbox_3d[6].z]
		Q8 = [msg.objects[i].bbox_3d[7].x,msg.objects[i].bbox_3d[7].y,msg.objects[i].bbox_3d[7].z]


		bb_2D = np.array([P1,P2,P3,P4])
		bb_3D = np.array([Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8])
		bb_6DoF =np.array([msg.objects[i].bbox_3d[1].x,msg.objects[i].bbox_3d[7].x,msg.objects[i].bbox_3d[1].y,
		msg.objects[i].bbox_3d[7].y,msg.objects[i].bbox_3d[1].z,msg.objects[i].bbox_3d[7].z])# this order is not ideal, should be min max  not max min
		label = (str(msg.objects[i].label_id),str(msg.objects[i].confidence))  #  can be more, depending on what you need when vis2d
		position = [round(msg.objects[i].position.x,2),round(msg.objects[i].position.y,2),round(msg.objects[i].position.z,2)]
		string_content ="id:"+ str(msg.objects[i].label_id)+" label:"+str(msg.objects[i].label)+' pos:'+str(position)+' conf:'+ str(msg.objects[i].confidence)+' State:'+ str(msg.objects[i].tracking_state)+" 2D_BB:"+str(bb_2D)
 



		array_2D_vis[i] = bb_2D
		array_3D_full[i] = bb_3D
		array_3D_vis[i] = bb_6DoF
		list_vis_label.append(label)
		list_full_info.append(string_content)
		dict_everything4frame[msg.objects[i].label_id] = string_content
 


 
    	f2.write(str(dict_everything4frame)+'\n')
    	dict_2D_vis[int(str(time))/1000] = array_2D_vis
    	dict_3D_vis[str(int(str(time))/1000)] = array_3D_vis #   we change the key of dict to str not int! just for this npy/pkl file, this is because the npz file only allow string key.
   	dict_3D_full[int(str(time))/1000] = array_3D_full
    	dict_vislabel[int(str(time))/1000] = list_vis_label
   	dict_fullinfo[int(str(time))/1000] = list_full_info
 

 
	
        rospy.sleep( 0.001)


def main():
    rospy.init_node('obj_det')
  
 
#object detection data
    obj_det_topic = "/zed2/zed_nodelet/obj_det/objects"


 
 

    rospy.Subscriber(obj_det_topic, Objects, callback)
    # Spin until ctrl + c
    rospy.spin()

 

if __name__ == '__main__':

 

    f = open("/media/tud-jxavier/SSD/data/zed/timestamps/timestamp_OD.txt",'a')
    f.truncate(0) 
    f2 = open("/media/tud-jxavier/SSD/data/zed/obj_det.txt",'a')
    f2.truncate(0)
    dict_2D_vis = {}
    dict_3D_vis = {}
    dict_3D_full = {}
    dict_vislabel = {}
    dict_fullinfo = {}
 
    


    main()

    with open("/media/tud-jxavier/SSD/data/zed/2D_vis_dict_array.pkl","wb") as tf:
	pickle.dump(dict_2D_vis,tf)
    with open("/media/tud-jxavier/SSD/data/zed/3D_vis_dict_array.pkl","wb") as tf:
	pickle.dump(dict_3D_vis,tf)
    with open("/media/tud-jxavier/SSD/data/zed/3D_full_array.pkl","wb") as tf:
	pickle.dump(dict_3D_full,tf)
    with open("/media/tud-jxavier/SSD/data/zed/dict_vislabel_list.pkl","wb") as tf:
	pickle.dump(dict_vislabel,tf)
    with open("/media/tud-jxavier/SSD/data/zed/dict_fullinfo_list.pkl","wb") as tf:
	pickle.dump(dict_fullinfo,tf)


    path_2d =  "/media/tud-jxavier/SSD/data/zed/2D_vis_dict_array.npy"
    np.save(path_2d,dict_2D_vis)
    path_3d =  "/media/tud-jxavier/SSD/data/zed/3D_vis_dict_array.npy"


    path_3dz =  "/media/tud-jxavier/SSD/data/zed/3D_vis_dict_array.npz"
    np.save(path_3d,dict_3D_vis)


    np.savez(path_3dz,**dict_3D_vis) #  this is the way we save npz file; we need npz for 3D visulisation;  the cpp file will only process this npz file
    path_3d_full =  "/media/tud-jxavier/SSD/data/zed/3D_full_array.npy"
    np.save(path_3d_full,dict_3D_full)
    path_vislabel =  "/media/tud-jxavier/SSD/data/zed/dict_vislabel_list.npy"
    np.save(path_vislabel, dict_vislabel)
    path_fullinfo =  "/media/tud-jxavier/SSD/data/zed/dict_fullinfo_list.npy"
    np.save(path_fullinfo,dict_fullinfo)

 
    f2.close()
    f.close()
    
