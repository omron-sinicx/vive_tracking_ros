cmake_minimum_required(VERSION 2.8.3)
project(vive_tracking_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  std_msgs
  tf_conversions
)

catkin_python_setup()

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(CATKIN_DEPENDS
  std_msgs
  sensor_msgs
  geometry_msgs
)
