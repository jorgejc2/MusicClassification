project(MusicClassification CXX CUDA)

pybind11_add_module(dsp_module dsp.pybind.cu)

# target_include_directories(dsp_module
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
# )

target_link_libraries(dsp_module PUBLIC dsp)

set_target_properties(dsp_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")