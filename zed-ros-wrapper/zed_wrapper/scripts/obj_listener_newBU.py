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
 


    f = open("/media/tud-jxavier/SSD/data/zed/timestamp_OD.txt", 'a')
    f.truncate(0) 
    f2 = open("/media/tud-jxavier/SSD/data/zed/obj_det.txt", 'a')
    f2.truncate(0)
    f3 = open("/media/tud-jxavier/SSD/data/zed/2D_vis.txt", 'a')
    f3.truncate(0) 
    f4 = open("/media/tud-jxavier/SSD/data/zed/label_2D_vis.txt", 'a')
    f4.truncate(0)     
    f5 = open("/media/tud-jxavier/SSD/data/zed/3D_vis.txt", 'a')
    f5.truncate(0)









	f.write(str(int(str(time))/1000)+' '+ str(date)+"\n")# only time stamp for alighment
	
 
 
	
	#obj detection
	f2.write(str(int(str(time))/1000)+' '+ str(date)+' totalNUM_objects: '+str(len(msg.objects))+" " )
	num4objs = len(msg.objects)
	#BB2d_array = np.zeros((4*num4objs,3))
	#BB3d_array = np.zeros((8*num4objs,3))
        list_2D = []
        list_3D = []
	list_label_ID = []
        list_pos = []
        list_conf = []
        list_state = []
	dict_everything4frame = {}
	list_detailsStr = []
	aabbs_6DoF = []
	labels = []
	
	
    	for i in range(len(msg.objects)):
    
       	 	if(msg.objects[i].label_id == -1):
           	    continue;


		#f2.write('label&ID:'+str(msg.objects[i].label)+" "+str(msg.objects[i].label_id)+' pos:'+str(position)+' conf:'+ str(msg.objects[i].confidence)+' State:'+ str(msg.objects[i].tracking_state)+"  "  )#+' linear velocity:['+str(msg.objects[i].linear_vel.x)+' '+str(msg.objects[i].linear_vel.y)+' '+str(msg.objects[i].linear_vel.z)+'] ')
		#f2.write("\n"+"2D BB xyz for point 1: "+str(mag.objects[i].bbox_2d[1].x)+" "+str(mag.objects[i].bbox_2d[1].y)+" "+str(mag.objects[i].bbox_2d[1].z))
		
		#grid_2D = np.ones((4,3))

		#(xmin, ymin, xmax, ymax) total 4 number to fit the visulization, we just need the four number
		P1 =msg.objects[i].bbox_2d[0].x #[msg.objects[i].bbox_2d[0].x,msg.objects[i].bbox_2d[0].y]#,msg.objects[i].bbox_2d[0].z]
		P2 =msg.objects[i].bbox_2d[0].y#[msg.objects[i].bbox_2d[1].x,msg.objects[i].bbox_2d[1].y]#,msg.objects[i].bbox_2d[1].z]
		P3 =msg.objects[i].bbox_2d[2].x#[msg.objects[i].bbox_2d[2].x,msg.objects[i].bbox_2d[2].y]#msg.objects[i].bbox_2d[2].z]
		P4 =msg.objects[i].bbox_2d[2].y#[msg.objects[i].bbox_2d[3].x,msg.objects[i].bbox_2d[3].y]#msg.objects[i].bbox_2d[3].z]

		#P11 = [msg.objects[i].bbox_2d[0].x,msg.objects[i].bbox_2d[0].y,msg.objects[i].bbox_2d[0].z]
		#P12 = [msg.objects[i].bbox_2d[1].x,msg.objects[i].bbox_2d[1].y,msg.objects[i].bbox_2d[1].z]
		#P13 = [msg.objects[i].bbox_2d[2].x,msg.objects[i].bbox_2d[2].y,msg.objects[i].bbox_2d[2].z]
		#P14 = [msg.objects[i].bbox_2d[3].x,msg.objects[i].bbox_2d[3].y,msg.objects[i].bbox_2d[3].z]


		#total 8*3 xyz = 24 number
		Q1 = [msg.objects[i].bbox_3d[0].x,msg.objects[i].bbox_3d[0].y,msg.objects[i].bbox_3d[0].z]
		Q2 = [msg.objects[i].bbox_3d[1].x,msg.objects[i].bbox_3d[1].y,msg.objects[i].bbox_3d[1].z]
		Q3 = [msg.objects[i].bbox_3d[2].x,msg.objects[i].bbox_3d[2].y,msg.objects[i].bbox_3d[2].z]
		Q4 = [msg.objects[i].bbox_3d[3].x,msg.objects[i].bbox_3d[3].y,msg.objects[i].bbox_3d[3].z]
		Q5 = [msg.objects[i].bbox_3d[4].x,msg.objects[i].bbox_3d[4].y,msg.objects[i].bbox_3d[4].z]
		Q6 = [msg.objects[i].bbox_3d[5].x,msg.objects[i].bbox_3d[5].y,msg.objects[i].bbox_3d[5].z]
		Q7 = [msg.objects[i].bbox_3d[6].x,msg.objects[i].bbox_3d[6].y,msg.objects[i].bbox_3d[6].z]
		Q8 = [msg.objects[i].bbox_3d[7].x,msg.objects[i].bbox_3d[7].y,msg.objects[i].bbox_3d[7].z]
		bb_2D = [P1,P2,P3,P4]
		bb_3D = [Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8]
		list_2D.append(bb_2D)
                list_3D.append(bb_3D)
		grid_2D = np.array([P1,P2,P3,P4]).reshape(1,4)
		grid_3D = np.array([Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8]).reshape(8,3)

		position = [round(msg.objects[i].position.x,2),round(msg.objects[i].position.y,2),round(msg.objects[i].position.z,2)]
		#print(P11,P12,P13,P14)
		string_content ="id:"+ str(msg.objects[i].label_id)+" label:"+str(msg.objects[i].label)+' pos:'+str(position)+' conf:'+ str(msg.objects[i].confidence)+' State:'+ str(msg.objects[i].tracking_state)+" 2D_BB:"+str(bb_2D)
		consise_label = "id:"+ str(msg.objects[i].label_id)+' conf:'+ str(msg.objects[i].confidence)

		string_identity = str(msg.objects[i].label_id)
		dict_everything4frame[string_identity] = string_content
		list_detailsStr.append(consise_label)



		aabb_6DoF =[msg.objects[i].bbox_3d[1].x,msg.objects[i].bbox_3d[7].x,msg.objects[i].bbox_3d[1].y,
			    msg.objects[i].bbox_3d[7].y,msg.objects[i].bbox_3d[1].z,msg.objects[i].bbox_3d[7].z]
		aabbs_6DoF.append(aabb_6DoF)
		label = (str(msg.objects[i].label_id),str(position),str(msg.objects[i].confidence))  #  can be more, depending on what you need when vis2d
		labels.append(label)
		





		#print("2D_BB: "+str(P1)+str(P2)+str(P3)+str(P4)+"  ")
		#print(list_detailsStr)

		#print("grid : ",grid_2D)
		#print("pos: ",position )
		#print("2D BB xyz for point 1: "+str(msg.objects[i].bbox_2d[1].x)+" "+str(msg.objects[i].bbox_2d[1].y)+" "+str(msg.objects[i].bbox_2d[1].z))
    	f2.write(str(dict_everything4frame)+'\n')
        BB2d_array = np.array(list_2D).reshape(num4objs,4)
        BB3d_array = np.array(list_3D).reshape(num4objs,8,3)
	path_2d =  "/media/tud-jxavier/SSD/data/zed/objects_2dBB/"+str(int(str(time))/1000)+".npy"
	np.save(path_2d,BB2d_array)
	path_3d =  "/media/tud-jxavier/SSD/data/zed/objects_3dBB/"+str(int(str(time))/1000)+".npy"
	np.save(path_3d,BB3d_array)
	path_list = "/media/tud-jxavier/SSD/data/zed/objects_details/"+str(int(str(time))/1000)+".npy"




#maybe should write in a txt not a dir of .txt files
	np.save(path_list,list_detailsStr)
	path_AABB = "/media/tud-jxavier/SSD/data/zed/objects_AABB/"+str(int(str(time))/1000)+".npy"
	np.save(path_AABB,aabbs_6DoF)
    	path_labels = "/media/tud-jxavier/SSD/data/zed/objects_labels/"+str(int(str(time))/1000)+".npy"
	np.save(path_labels,labels)
#   list with 4 elements, each elements has x,y,z	
#+' 2D bounding box:['+str(msg.objects[i].bbox_2d.x)+' '+str(msg.objects[i].bbox_2d.y)+' '+str(msg.objects[i].bbox_2d.z)+'] '




	
        rospy.sleep( 0.001)


def main():
    rospy.init_node('obj_det')
  
 
#object detection data
    obj_det_topic = "/zed2/zed_nodelet/obj_det/objects"


 
 

    rospy.Subscriber(obj_det_topic, Objects, callback)
    # Spin until ctrl + c
    rospy.spin()

    





    # Set up your subscriber and define its callback
   # rospy.Subscriber(image_topic_left, Image, image_callback)
    # Spin until ctrl + c
    #rospy.spin()

if __name__ == '__main__':

    files2D = glob.glob('/media/tud-jxavier/SSD/data/zed/objects_2dBB/*')
    files3D = glob.glob('/media/tud-jxavier/SSD/data/zed/objects_3dBB/*')
    filesDetails = glob.glob('/media/tud-jxavier/SSD/data/zed/objects_details/*')
    filesLabels = glob.glob('/media/tud-jxavier/SSD/data/zed/objects_labels/*')
    filesAABB = glob.glob('/media/tud-jxavier/SSD/data/zed/objects_AABB/*')
    for i in files2D:
        os.remove(i)
    for i in files3D:
        os.remove(i)
    for i in filesDetails:
        os.remove(i)
    for i in filesLabels:
        os.remove(i)
    for i in filesAABB:
        os.remove(i)

    f = open("/media/tud-jxavier/SSD/data/zed/timestamp_OD.txt", 'a')
    f.truncate(0) 
    f2 = open("/media/tud-jxavier/SSD/data/zed/obj_det.txt", 'a')
    f2.truncate(0)
    f3 = open("/media/tud-jxavier/SSD/data/zed/2D_vis.txt", 'a')
    f3.truncate(0) 
    f4 = open("/media/tud-jxavier/SSD/data/zed/label_2D_vis.txt", 'a')
    f4.truncate(0)     
    f5 = open("/media/tud-jxavier/SSD/data/zed/3D_vis.txt", 'a')
    f5.truncate(0)
    


    main()
    f5.close() 
    f4.close()
    f3.close() 
    f2.close()
    f.close()
    
