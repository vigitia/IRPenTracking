cmake_minimum_required(VERSION 3.5)
project(vigitia_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArucoMarker.msg"
  "msg/ArucoList.msg"
  "msg/DetectedObject.msg"
  "msg/DetectedObjectsList.msg"
  "msg/Page.msg"
  "msg/PageList.msg"
  DEPENDENCIES std_msgs geometry_msgs
  ADD_LINTER_TESTS
 )

ament_export_dependencies(rosidl_default_runtime)

ament_package()
