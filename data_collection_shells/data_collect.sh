#!/bin/bash

rosnode kill -a 

sudo systemctl restart roverrobotics.service
sleep 5
if systemctl is-active -q roverrobotics.service ; then
	sleep 1
else
	echo "systemctl is not active, exit...."
	exit 
fi


if systemctl status roverrobotics.service | grep "ERROR" 
  then echo "error found in restarting, exit...."
	exit
fi
#roslaunch rr_openrover_driver starterkit_bringup.launch 

 
echo "####Start to clean the pcd dirs...####"
python /home/tud-jxavier/catkin_ws/src/zed-ros-wrapper/zed_wrapper/scripts/pointcloud_dir_cleaner_zed.py
python /home/tud-jxavier/catkin_ws/src/ros_rslidar/rslidar_pointcloud/scripts/pointcloud_dir_cleaner_rslidar.py
echo "####Done with cleaning the pcd dirs!####"
echo "####Start data collection... please ctrl+c when you want to finish the collection.####"


roslaunch rslidar_sdk data_collection.launch &
sleep 30
rosnode kill -a

echo "####Finished data collection!####" 
echo "####Start tsp generation..####"
#python pcd_tsp_generator.launch


python /home/tud-jxavier/catkin_ws/src/zed-ros-wrapper/zed_wrapper/scripts/pointcloud_tsp_generator_zed.py
python /home/tud-jxavier/catkin_ws/src/ros_rslidar/rslidar_pointcloud/scripts/pointcloud_tsp_generator_rslidar.py
echo "####Finished tsp generation!####"
echo "####Start to copy the dataset named with the current time...####"

NOW=`date '+%F_%H:%M:%S'`
cp -R /media/tud-jxavier/SSD/data  /media/tud-jxavier/SSD/Datasets/data_$NOW

echo "####Done! you can start next round of data collection!####"


