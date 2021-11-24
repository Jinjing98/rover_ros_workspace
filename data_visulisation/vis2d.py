#!/usr/bin/python


import sys
print('sys version is :',sys.version)
# by manual import this, we don't have to install any package.
sys.path.insert(0, '/media/tud-jxavier/SSD/data/zed/visulisation/bbox-visualizer/')

import bbox_visualizer as bbv
import cv2
import numpy as np
import json
import pickle
import sys
 
coord_dict = np.load('/media/tud-jxavier/SSD/data/zed/object_detection_data/2D_vis_dict_array.npz')
label_dict = np.load('/media/tud-jxavier/SSD/data/zed/object_detection_data/dict_vislabel_list.npz')

# make sure there are related matched img and BB(OD) for the timestamp
def vis2dBB(timestamp4BBNlabel):
	img_path = "/media/tud-jxavier/SSD/data/zed/imgLeft/"+str(timestamp4BBNlabel)+".jpeg"
	BBs_array =  coord_dict[str(timestamp4BBNlabel)]#.astype(np.int32)
	labels_list = label_dict[str(timestamp4BBNlabel)]




	img = cv2.imread(img_path)
	num4objs = BBs_array.shape[0]
#  drawing 2D BB
	for i in range(num4objs):
		BB = list(BBs_array[i].astype(np.int32))
		print('BB:',BB)
		label = labels_list[i]
		label_string = str(label[0].decode('utf-8'))+" "+str(label[1].decode('utf-8'))
		#print(label)
		#print(str(label[0].decode('utf-8'))+"  "+str(label[1].decode('utf-8')))
 
 

		new = bbv.draw_rectangle(img, BB,is_opaque=True)
		new = bbv.add_label(new, label_string, BB, top=True )
		img = new
	cv2.imshow(" ",new)
	cv2.waitKey(0)

# make sure "perfect match" of timestamps for BB(OD) and left_img! 
def visallStream(alighed_OD_IMG):
	with open(alighed_OD_IMG) as f:
		line = f.readline()
		while line:
			line = f.readline()[:16]
			vis2dBB(int(line))
			print(line)
 


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("usage: python3 vis.py path_of_alighed_timestamps_of_od_img ")
        print('usage: python3 vis.py timestamp_of_chosen_frame    the chosen timestamp should have corrosponding left image and object detection information ')
    elif sys.argv[1][-3:]=='txt':
        alighed_OD_IMG = sys.argv[1]
        visallStream(alighed_OD_IMG)
    else:
        vis2dBB(int(sys.argv[1]))
 
 

	







