#!/bin/bash

rosnode kill -a 
echo "####Please key in password: pass####"
systemctl restart roverrobotics.service 
echo "####Start to clean the pcd dirs... please ctrl+c for TWO times after several sec...####"
rosrun rslidar_pointcloud pointcloud_dir_cleaner_rslidar.py 
rosrun zed_wrapper pointcloud_dir_cleaner_zed.py
echo "####Done with cleaning the pcd dirs!####"
echo "####Start data collection... please ctrl+c when you want to finish the collection.####"


roslaunch rslidar_sdk data_collection.launch 

echo "####Finished data collection!####" 
echo "####Start tsp generation...please ctrl+c after several seconds...####"
roslaunch rslidar_sdk pcd_tsp_generator.launch
echo "####Finished tsp generation!####"
echo "####Start to copy the dataset named with the current time...####"

NOW=`date '+%F_%H:%M:%S'`
cp -R /media/tud-jxavier/SSD/data  /media/tud-jxavier/SSD/Datasets/data_$NOW

echo "####Done! you can start next round of data collection!####"

