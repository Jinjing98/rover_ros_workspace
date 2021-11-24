#cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(vis3d_BB)

 


 

find_package(PCL  REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS} /usr/local/)#include_directories(/usr/local/)
link_directories(/usr/local/lib)#  libcnpy is under such path
link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})
#add_executable (vis3d vis3d.cpp)
#target_link_libraries (vis3d ${PCL_LIBRARIES})




add_executable(vis3d vis3d.cpp)
target_link_libraries(vis3d cnpy ${PCL_LIBRARIES})

