cmake_minimum_required(VERSION 3.0.2)
project(mono_following)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  tf
  rospy
  roscpp
  std_msgs
  cv_bridge
  geometry_msgs
  tf_conversions
  image_transport
  eigen_conversions
  message_generation
  mono_tracking
  actionlib_msgs
)

# catkin_python_setup()

# find_package(OpenCV 3.4 REQUIRED)
# find_package(Eigen3 REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Box.msg
  Samples.msg
  Target.msg
  Descriptor.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  extractDescriptors.srv
  updateDescriptors.srv
  classifyTarget.srv
)

## Generate actions in the 'action' folder
add_action_files(
  FILES
  updateExtractor.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

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
#  LIBRARIES mono_following
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs
)



# add_executable(mono_following_node
#   src/descriminator.cpp
#   src/initial_state.cpp
#   src/initial_training_state.cpp
#   src/mono_following_node.cpp
#   src/reid_state.cpp
#   src/tracking_state.cpp
# )

# target_link_libraries(mono_following_node
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBS}
# )
# add_dependencies(mono_following_node mono_following_gencpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# catkin_install_python(PROGRAMS scripts/mono_following/mono_following_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )