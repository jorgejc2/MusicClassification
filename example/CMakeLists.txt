project(MusicClassification CXX CUDA)

# temporarily discarding stft file to compile a Pybind module version of dsp isntead

add_executable(wav_to_arr basic_wav_to_array/wav_to_arr.cpp)
# add_executable(stft basic_stft/stft.cu)

find_package(NumCpp)

target_link_libraries(wav_to_arr 
    PRIVATE
    wavreader
)

# set_target_properties(stft 
#     PROPERTIES 
#     CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(stft
#     PRIVATE
#     wavreader
#     dsp
#     NumCpp::NumCpp
# )

# set_target_properties(stft PROPERTIES CUDA_ARCHITECTURES "35;50;72")

