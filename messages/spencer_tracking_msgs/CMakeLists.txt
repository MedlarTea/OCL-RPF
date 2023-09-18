cmake_minimum_required(VERSION 2.8.3)
project(spencer_tracking_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

# Generate messages in the 'msg' folder
add_message_files(
   FILES
   BodyPartElm.msg
   Box.msg
   DetectedPerson.msg
   DetectedPersons.msg
   DetectedPerson2d.msg
   DetectedPersons2d.msg
   CompositeDetectedPerson.msg
   CompositeDetectedPersons.msg
   TargetPerson.msg
   TrackedPerson.msg
   TrackedPersons.msg
   TrackedPerson2d.msg
   TrackedPersons2d.msg
   TrackedGroup.msg
   TrackedGroups.msg
   ImmDebugInfo.msg
   ImmDebugInfos.msg
   TrackingTimingMetrics.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  GetPersonTrajectories.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs
   sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs message_runtime 
)
