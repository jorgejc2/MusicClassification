project(MusicClassification CXX CUDA)

# pybind11_add_module(dsp_module pybind_src/dsp.pybind.cu)
pybind11_add_module(dsp_module 
    dsp.module.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/dsp.pybind.cu
)
pybind11_add_module(wav_module 
    wav.module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/wav.pybind.cpp
)
pybind11_add_module(matrix_module 
    matrix.module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/matrix.pybind.cpp
)

# add_library(wavreader pybind_src/wav.pybind.cpp)
# add_library(dsp pybind_src/dsp.pybind.cu)
# add_library(matrix pybind_src/matrix.pybind.cpp)

target_link_libraries(wav_module PUBLIC wavreader)
target_link_libraries(matrix_module PUBLIC matrix)
target_link_libraries(dsp_module 
    PUBLIC 
    dsp
    # matrix_module
)

set_target_properties(dsp_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(wav_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")