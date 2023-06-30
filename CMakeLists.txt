cmake_minimum_required(VERSION 2.8.3)
project(vive_tracking_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  sensor_msgs
  std_msgs
  tf_conversions
)

catkin_python_setup()

include_directories(${catkin_INCLUDE_DIRS})

add_message_files(DIRECTORY msg)


generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  sensor_msgs
  geometry_msgs
)
