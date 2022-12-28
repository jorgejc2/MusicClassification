cmake_minimum_required(VERSION 3.15)
project(MusicClassification)

# want this flag so that debug symbols can be generated for gdb
# NOTE: adding compilation flags should always happen before adding subdirectories
add_compile_options(-g)

add_subdirectory(include)
add_subdirectory(src)

set(CMAKE_CXX_STANDARD 17)

option (BUILD_EXAMPLES "Whether or not to build examples"
ON)

if (BUILD_EXAMPLES)
    message(STATUS "Building examples...")
    add_subdirectory(example)
endif()