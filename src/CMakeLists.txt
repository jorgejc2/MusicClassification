project(MusicClassification CXX CUDA)

add_library(wavreader wavread.cpp)
add_library(dsp dsp.cu)

# so that it can be read in its pybind module counterpart
set_target_properties(dsp 
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
)

set_target_properties(wavreader
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
)

target_include_directories(wavreader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(dsp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

set_target_properties(wavreader PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(dsp PROPERTIES CUDA_ARCHITECTURES "35;50;72" PREFIX "")