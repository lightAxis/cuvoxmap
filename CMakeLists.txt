cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(add_compile_warnings)
include(parse_git_tag)
include(detect_nvcc_version)

cmake_policy(SET CMP0104 NEW)

project(
    cuvoxmap
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    DESCRIPTION "A Lightweight C++ Library for grid map in robotics"
    LANGUAGES CXX C CUDA
)

## compile options
set(CUVOXMAP_USING_CUDA OFF)

## enabling cuda
if(CUVOXMAP_USING_CUDA)
    find_package(CUDA REQUIRED)
    detect_nvcc_version(NVCC_VERSION REQUIRED 12.3)
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

if(NOT BUILD_TESTING)
    set(BUILD_TESTING ON)
endif()

include_directories(include)

if(CUVOXMAP_USING_CUDA)
    add_library(cuvoxmap 
        src/base/MapAllocator.cu 
        src/base/MapImpl.cu)
else()
    add_library(cuvoxmap 
        src/base/MapAllocator.cpp 
        src/base/MapImpl.cpp)
endif()
# if(CUVOXMAP_USING_CUDA)
#     target_compile_definitions(cuvoxmap PUBLIC CUVOXMAP_USING_CUDA)
# endif()
add_library(cuvoxmap::main ALIAS cuvoxmap)

add_subdirectory(test)


# add_library(cuvoxmap)
# add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS cuvoxmap)