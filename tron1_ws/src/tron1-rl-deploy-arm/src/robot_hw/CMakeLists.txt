# Copyright information
#
# © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(robot_hw)

# --- Optional: ARX5 SDK integration (real arm control) ---
option(USE_ARX5_SDK "Enable ARX5 SDK arm integration in PointfootHW (real robot)" OFF)
set(ARX5_SDK_DIR "" CACHE PATH "Path to arx5-sdk root (contains include/, src/, lib/)")

# Enable generation of compile commands database (used by some code editors)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
# Find and specify required catkin packages and components
find_package(catkin REQUIRED 
  COMPONENTS
  roscpp
  geometry_msgs
  pointfoot_sdk_lowlevel
  robot_common
  robot_description
  hardware_interface
  controller_manager
  urdf
  kdl_parser
  # quest_msg
  airbot_msgs
  diagnostic_msgs
  std_msgs
  airbot_sdk # airbot sdk
)

set(AIRBOT_SDK_LIB_FILES  libairbot_play.so
        libairbot_play.so.2.9
        libairbot_play.so.2.9.0)

# Specify catkin-specific package configuration
catkin_package(
  INCLUDE_DIRS
  include
  LIBRARIES
  ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  geometry_msgs
  pointfoot_sdk_lowlevel
  robot_common
  robot_description
  hardware_interface
  controller_manager
  urdf
  kdl_parser
  # quest_msg
  airbot_msgs
  diagnostic_msgs
  std_msgs
  airbot_sdk # airbot sdk
)

find_package(urdfdom REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

if (USE_ARX5_SDK)
  add_definitions(-DUSE_ARX5_SDK)
  # We intentionally only depend on the legacy SocketCAN (no-soem) subset:
  #   - headers: include/hardware/arx_can.h
  #   - runtime: lib/x86_64/libhardware.so or lib/aarch64/libhardware.so
  if (NOT EXISTS "${ARX5_SDK_DIR}/include/hardware/arx_can.h")
    message(FATAL_ERROR "USE_ARX5_SDK=ON but ARX5_SDK_DIR is invalid. Set -DARX5_SDK_DIR=/path/to/arx5-sdk")
  endif()

  # Choose prebuilt binary libs for hardware/solver by architecture.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(ARX5_LIB_DIR ${ARX5_SDK_DIR}/lib/aarch64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(ARX5_LIB_DIR ${ARX5_SDK_DIR}/lib/x86_64)
  else()
    message(FATAL_ERROR "Unknown architecture for ARX5 SDK: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

endif()

# Include directories for header files
include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

set(POINTFOOT_NODE_SRCS
  ${PROJECT_SOURCE_DIR}/src/RobotHW.cpp
  ${PROJECT_SOURCE_DIR}/src/RobotHWLoop.cpp
  ${PROJECT_SOURCE_DIR}/src/PointfootHW.cpp
  ${PROJECT_SOURCE_DIR}/src/pointfootNode.cpp
)
if (USE_ARX5_SDK)
  list(APPEND POINTFOOT_NODE_SRCS
    ${PROJECT_SOURCE_DIR}/src/Arx5SocketCanArm.cpp
  )
endif()

add_executable(pointfoot_node ${POINTFOOT_NODE_SRCS})

target_link_libraries(pointfoot_node
        ${catkin_LIBRARIES}  # Link with catkin libraries
        ${urdfdom_LIBRARIES}
        pointfoot_sdk_lowlevel
        pthread
        ${AIRBOT_SDK_LIB_FILES}
        ${YAML_CPP_LIBRARIES}
)

if (USE_ARX5_SDK)
  target_include_directories(pointfoot_node PRIVATE ${ARX5_SDK_DIR}/include)
  target_link_libraries(pointfoot_node
    ${ARX5_LIB_DIR}/libhardware.so
    Threads::Threads
  )
endif()

# Mark executables and/or libraries for installation
install(TARGETS pointfoot_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(DIRECTORY include/. DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# Mark resource files for installation
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
