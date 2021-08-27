#include <sstream>
#include <boost/filesystem.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "pcl/io/io.h"
#include "pcl/io/pcd_io.h"
#include "pcl_ros/transforms.h"
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

typedef sensor_msgs::PointCloud2 PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef PointCloud::ConstPtr PointCloudConstPtr;

/* ---[ */
int
  main (int argc, char** argv)
{
  ros::init (argc, argv, "bag_to_pcd");
  if (argc < 4)
  {
    std::cerr << "Syntax is: " << argv[0] << " <file_in.bag> <topic> <output_directory>" << std::endl;
    std::cerr << "Example: " << argv[0] << " data.bag /laser_tilt_cloud ./pointclouds" << std::endl;
    return (-1);
  }

  // TF
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;

  rosbag::Bag bag;
  rosbag::View view;
  rosbag::View::iterator view_it;

  try
  {
    bag.open (argv[1], rosbag::bagmode::Read);
  } 
  catch (rosbag::BagException) 
  {
    std::cerr << "Error opening file " << argv[1] << std::endl;
    return (-1);
  }

  view.addQuery (bag, rosbag::TypeQuery ("sensor_msgs/PointCloud2"));
  view.addQuery (bag, rosbag::TypeQuery ("tf/tfMessage"));
  view_it = view.begin ();

  std::string output_dir = std::string (argv[3]);
  boost::filesystem::path outpath (output_dir);
  if (!boost::filesystem::exists (outpath))
  {
    if (!boost::filesystem::create_directories (outpath))
    {
      std::cerr << "Error creating directory " << output_dir << std::endl;
      return (-1);
    }
    std::cerr << "Creating directory " << output_dir << std::endl;
  }

  // Add the PointCloud2 handler
  std::cerr << "Saving recorded sensor_msgs::PointCloud2 messages on topic " << argv[2] << " to " << output_dir << std::endl;

  PointCloud cloud_t;
  ros::Duration r (0.001);
  // Loop over the whole bag file
  while (view_it != view.end ())
  {
    // Handle TF messages first
    tf::tfMessage::ConstPtr tf = view_it->instantiate<tf::tfMessage> ();
    if (tf != NULL)
    {
      tf_broadcaster.sendTransform (tf->transforms);
      ros::spinOnce ();
      r.sleep ();
    }
    else
    {
      // Get the PointCloud2 message
      PointCloudConstPtr cloud = view_it->instantiate<PointCloud> ();
      if (cloud == NULL)
      {
        ++view_it;
        continue;
      }
      // Transform it
      pcl_ros::transformPointCloud ("/base_link", *cloud, cloud_t, tf_listener);

     // std::cerr << "Got " << cloud_t.width * cloud_t.height << " data points in frame " << cloud_t.header.frame_id << " with the following fields: " << pcl::getFieldsList (cloud_t) << std::endl;

   
//std::to_string((long long)((atof(ss.str ().c_str ( ))*1000000)) 
 //     ss << output_dir << "/" << cloud_t.header.stamp << ".pcd";
//(long long)((atof(ss.str ().c_str ( ))*1000000)


      std::stringstream ss;
      ss <<cloud_t.header.stamp;// << ".pcd";
      ROS_INFO ("Data saved to %s",   std::to_string((atof(ss.str ().c_str ( ))*1000000)) );
 

      pcl::io::savePCDFile ( output_dir+std::to_string((long long)((atof(ss.str ().c_str ( ))*1000000)) )+".pcd", cloud_t, Eigen::Vector4f::Zero (),
                            Eigen::Quaternionf::Identity (), true);

 
    }
    // Increment the iterator
    ++view_it;
  }

  return (0);
}
