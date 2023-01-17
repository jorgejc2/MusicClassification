project(MusicClassification CXX CUDA)

add_library(wavreader wavread.cpp)
add_library(dsp dsp.cu)
pybind11_add_module(dsp_module dsp_pybind.pybind.cpp)

set_target_properties(dsp 
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(dsp_module PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_include_directories(wavreader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# target_link_libraries(dsp ${PYTHON_LIBRARIES})
target_include_directories(dsp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    # ${CMAKE_CURRENT_SOURCE_DIR}/../pybind11/include
)

# target_link_libraries(dsp_module ${PYTHON_LIBRARIES})
target_include_directories(dsp_module
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../pybind11/include
)

# add_library(MusicClassification MODULE dsp_pybind.pybind.cpp)

set_target_properties(wavreader PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set_target_properties(dsp PROPERTIES CUDA_ARCHITECTURES "35;50;72")
# set_target_properties(dsp_module PROPERTIES CUDA_ARCHITECTURES "35;50;72")

# target_link_libraries(dsp_module PRIVATE pybind11::module)