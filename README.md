The repo include all the src files of catkin_ws on Jeston. 

note: The two sub directory `data_collection_shells` and `data_visulisation` are not ros packages. They are for simple data collection and post processing visulisation. For completeness, I also put them under this repo. 

About data collection:

there are three tasks we want to achieve:
1) online visulisation of zed pointcloud, lidar pointcloud, zed bounding box in Rviz.(no data collection)
2) collect data. each round of recording are saved and named with its recording time.
3) based on the collect data, do offline visualisation. we can do the visuilisation of point cloud and/or bounding box in both rviz and non ros environment.

The details of how to achieve the above three tasks please refer to the howto pdf file under `data_collection_shells`.
