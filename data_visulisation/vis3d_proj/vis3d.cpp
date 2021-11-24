#include <iostream>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <vtkAutoInit.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include"cnpy.h"
#include <fstream>

 
 


void aabb(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,double* arr_ptr,int num4objs)//Point cloud AABB bounding box
{
	pcl::MomentOfInertiaEstimation <pcl::PointXYZRGB> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();

	std::vector <float> moment_of_inertia;
	std::vector <float> eccentricity;

	pcl::PointXYZRGB min_point_AABB;//AABB bounding box
	pcl::PointXYZRGB max_point_AABB;
	max_point_AABB.x,max_point_AABB.y,max_point_AABB.z = 1.02719069, -0.40725103, -0.45342737 ;
	min_point_AABB.x,max_point_AABB.y,max_point_AABB.z = 0.43709287, 0.17508557 ,0.26943752;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getMomentOfInertia(moment_of_inertia);
	feature_extractor.getEccentricity(eccentricity);
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

	//Draw AABB bounding box
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(1,1,1);// 0 0 0
	//viewer->addCoordinateSystem(1.0);

	//pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGB> RandomColor(cloud);//Set random color
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud,rgb, "points");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "points");

	//viewer->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");
//1.02719069,0.4719069, -0.40725103,0.2719069, -0.45342737 ,0.3719069, 
	//viewer->addCube(0.4,1.0,-0.4,0.2,-0.5,0.3,  1.0f, 0.0f, 0.0f, "AABB" , 0);

	for (int i = 0;i<num4objs;i++){

// std::cout<<arr_ptr[0]<<" "<<arr_ptr[1]<<arr_ptr[2]<<num4objs<<" "<<ncols<<endl;
std::vector<float> BB;
BB.push_back(arr_ptr[0+i*6]);
BB.push_back(arr_ptr[1+i*6]);
BB.push_back(arr_ptr[2+i*6]);
BB.push_back(arr_ptr[3+i*6]);
BB.push_back(arr_ptr[4+i*6]);
BB.push_back(arr_ptr[5+i*6]);


viewer->addCube(BB[1],BB[0],BB[3],BB[2],BB[5],BB[4],0.0f, 0.5f, 0.0f, std::to_string(i) , 0);//  the order of minx maxx miny maxy ...
viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.2, std::to_string(i) );

}




	
	//viewer->addCube(0.4,1.0,-0.4,0.2,-0.5,0.3,0.0f, 0.5f, 0.0f, "cube" , 0);
	//viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 		pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "BB");
 	

	
	//pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	//pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	//pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	//pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
	//view frames
	//viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
	//viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	//viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(0.001);//last for 1 sec
	}


}
using namespace std;
//"/media/tud-jxavier/SSD/testdata/1623863833660239.pcd" 
///media/tud-jxavier/SSD/data/zed/point_cloud/1623863833660239.pcd

//    1:MAX 7:MIN
//  1.83812976,  1.10650146,  0.00256082,    -0.77297974,  0.48850438,-0.76750904
//[msg.objects[i].bbox_3d[1].x,msg.objects[i].bbox_3d[7].x,msg.objects[i].bbox_3d[1].y,msg.objects[i].bbox_3d[7].y,msg.objects[i].bbox_3d[1].z,msg.objects[i].bbox_3d[7].z]


void aabb_stream(std::string path4tsps){


	
std::string fname = "/media/tud-jxavier/SSD/data/zed/object_detection_data/3D_vis_dict_array.npz";//  not need to input
cnpy::npz_t all_npys = cnpy::npz_load(fname);

    ifstream file( path4tsps, ios::in );
    string part1, part2,part3;
    string num1, num2;

    if( !file )
        cerr << "Cant open " << endl;

    while( file  >> num1 >>num2 >>  part1>> part2>>part3 )
    {
 
        cout << num1
        << " " << "loading point cloud ..." << endl;
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/media/tud-jxavier/SSD/data/zed/point_cloud/"+string(num1)+string(".pcd"), *cloud);


	cnpy::NpyArray arr = all_npys[num1];
	double* arr_ptr = arr.data<double>();
	int num4objs = arr.shape[0];	
	
	aabb(cloud,arr_ptr,num4objs);//  vis one frame of all the BB in th


    }

    file.close();







}

















int main(int argc, char* argv[]){


std::cout<<"usage: ./vis3d path_of_alighed_timestamps_of_pcd_od "<<endl;
std::cout<<"usage: ./vis3d timestamp_of_chosen_frame    the chosen timestamp should have corrosponding pcd file and object detection information "<<endl;



std::string s(argv[1]);
if (s.back()!='t'){
//std::cout<<argc<<endl;
//std::cout<<"argv "<<argv[1]<<endl;

std::string fname = "/media/tud-jxavier/SSD/data/zed/object_detection_data/3D_vis_dict_array.npz";//  not need to input
//cnpy::npz_t all_npys = cnpy::npz_load(fname);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/media/tud-jxavier/SSD/data/zed/point_cloud/"+string(argv[1])+string(".pcd"), *cloud);//  input tsp  argv[1]


cnpy::NpyArray arr = cnpy::npz_load(fname,argv[1]);//   input the tsp of this frame
 
//cnpy::NpyArray arr = all_npys[argv[1]];





double* arr_ptr = arr.data<double>();
int num4objs = arr.shape[0];//1   num of objs
int ncols = arr.shape[1];//6
aabb(cloud,arr_ptr,num4objs);


}else{


std::string path4tsps = argv[1];
aabb_stream(path4tsps);


}

 

 



 



 




 

return 0;


}

































