cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(add_compile_warnings)
include(parse_git_tag)
include(detect_nvcc_version)

# cmake_policy(SET CMP0104 NEW)

project(
    cuvoxmap
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    DESCRIPTION "A Lightweight C++ Library for grid map in robotics"
    LANGUAGES CXX C CUDA
)

## compile options
set(CUVOXMAP_USING_CUDA ON)
message(STATUS "testing " ${BUILD_TESTING})
if(NOT DEFINED BUILD_TESTING)
    message(STATUS "BUILD_TESTING is not defined, set to ON")
    set(BUILD_TESTING ON)
endif()

## check TEGRA system to use unified memory or not
if(EXISTS "/etc/nv_tegra_release")
  target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_ARCH_TEGRA)
  message("-- TEGRA detected, CUDA_ARCH_TEGRA is defined for unified memory")
endif()


## enabling cuda
if(CUVOXMAP_USING_CUDA)
    find_package(CUDA REQUIRED)
    detect_nvcc_version(NVCC_VERSION REQUIRED 12.3)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(STATUS "CUDA DETECTED : ARCH " ${CMAKE_CUDA_ARCHITECTURES} " VERSION " ${NVCC_VERSION})
endif()

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# using C++ 17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include_directories(include)

if(CUVOXMAP_USING_CUDA)
    add_library(cuvoxmap_gpu 
        src/base/MapAllocator.cu 
        src/base/MapImpl.cu
        src/cuvoxmap.cpp
        src/distance_transform.cpp
        src/distance_transform.cu
        src/utils/MovablePtr.cu)
endif()
add_library(cuvoxmap_cpu 
    src/base/MapAllocator.cpp 
    src/base/MapImpl.cpp
    src/cuvoxmap.cpp
    src/distance_transform.cpp
    src/cuvoxmap_gpuonly.cpp
    src/utils/MovablePtr.cpp)
# if(CUVOXMAP_USING_CUDA)
#     target_compile_definitions(cuvoxmap PUBLIC CUVOXMAP_USING_CUDA)
# endif()
add_library(cuvoxmap::cpu ALIAS cuvoxmap_cpu)
add_library(cuvoxmap::gpu ALIAS cuvoxmap_gpu)

## TESTING
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

################
##### ROS2 #####
################

# find packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
include_directories(ros2/include)

## ROS2 node
add_executable(cuvoxmap_node_cpu ros2/src/cuvoxmap_main.cpp)
add_executable(cuvoxmap_node_gpu ros2/src/cuvoxmap_main_gpu.cpp)
target_link_libraries(cuvoxmap_node_cpu cuvoxmap_cpu)
target_link_libraries(cuvoxmap_node_gpu cuvoxmap_gpu)
ament_target_dependencies(cuvoxmap_node_cpu rclcpp sensor_msgs)
ament_target_dependencies(cuvoxmap_node_gpu rclcpp sensor_msgs)

## install
install(DIRECTORY include DESTINATION include)
install(DIRECTORY ros2/include DESTINATION include)
install(DIRECTORY ros2/launch DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY ros2/param DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET )

install(
    TARGETS 
        cuvoxmap_cpu cuvoxmap_gpu
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    TARGETS 
        cuvoxmap_node_cpu cuvoxmap_node_gpu
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()


# add_library(cuvoxmap)
# add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS cuvoxmap)