cmake_minimum_required(VERSION 2.8.3)
project(vision)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  my_roscpp_library
  std_msgs
  image_transport
  sensor_msgs
)
find_package( OpenCV REQUIRED )
#find_package(OpenCV COMPONENTS core highgui video imgproc REQUIRED)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vision
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(video_cam src/video_cam.cpp)
target_link_libraries(video_cam ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(trash_detection src/trash_detection.cpp)
target_link_libraries(trash_detection ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(target_detection src/target_detection.cpp)
target_link_libraries(target_detection ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(camera_calibration src/camera_calibration.cpp)
target_link_libraries(camera_calibration ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(detection src/detection.cpp)
target_link_libraries(detection ${catkin_LIBRARIES} ${OpenCV_LIBS})

