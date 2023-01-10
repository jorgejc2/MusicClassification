cmake_minimum_required(VERSION 3.17)
project(MusicClassification CXX CUDA)


# # to enable CUDA
# enable_language(CUDA)
message(STATUS "Finding CUDA package")
find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

# want this flag so that debug symbols can be generated for gdb
# NOTE: adding compilation flags should always happen before adding subdirectories
# add_compile_options(-g)

message(STATUS "Adding subdirectories")
add_subdirectory(include)
add_subdirectory(src)

option (BUILD_EXAMPLES "Whether or not to build examples"
ON)

if (BUILD_EXAMPLES)
    message(STATUS "Building examples...")
    add_subdirectory(example)
endif()