project(MusicClassification CXX CUDA)

add_library(wavreader wavread.cpp)
add_library(dsp dsp.cu)

set_target_properties(dsp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(wavreader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(dsp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
