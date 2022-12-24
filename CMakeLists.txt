cmake_minimum_required(VERSION 3.15)
project(MusicClassification)

add_subdirectory(include)
add_subdirectory(src)

set(CMAKE_CXX_STANDARD 17)

option (BUILD_EXAMPLES "Whether or not to build examples"
ON)

if (BUILD_EXAMPLES)
    message(STATUS "Building examples...")
    add_subdirectory(example)
endif()