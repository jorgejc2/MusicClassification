project(MusicClassification CXX CUDA)

# leave this CMake responsible for generating source files dealing with Pybind11
# Unfortunately I cannot find how to link source files between different Pybin11 modules, 
# so the dsp_module has to recompile the matrix pybind source file
pybind11_add_module(dsp_module 
    dsp.module.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/dsp.pybind.cu
)
pybind11_add_module(wav_module 
    wav.module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/wav.pybind.cpp
)
pybind11_add_module(matrix_module 
    matrix.module.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/matrix.pybind.cpp
)

target_link_libraries(wav_module PUBLIC wavreader)
target_link_libraries(matrix_module PUBLIC matrix dsp)
target_link_libraries(dsp_module PUBLIC dsp)

set_target_properties(wav_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(matrix_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(dsp_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")