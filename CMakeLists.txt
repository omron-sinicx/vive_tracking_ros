cmake_minimum_required(VERSION 2.8.3)
project(vive_tracking_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_srvs
  sensor_msgs
  tf_conversions
)

catkin_python_setup()

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
