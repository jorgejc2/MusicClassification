project(MusicClassification CXX CUDA)

pybind11_add_module(dsp_module dsp.pybind.cu)
pybind11_add_module(wav_module wav.pybind.cpp)
pybind11_add_module(matrix_module matrix.pybind.cpp)

target_link_libraries(dsp_module PUBLIC dsp PUBLIC matrix)
target_link_libraries(wav_module PUBLIC wavreader)
target_link_libraries(matrix_module PUBLIC matrix)

set_target_properties(dsp_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(wav_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")