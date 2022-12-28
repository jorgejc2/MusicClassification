project(MusicClassification)

add_executable(wav_to_arr basic_wav_to_array/wav_to_arr.cpp)
add_executable(stft basic_stft/stft.cpp)

find_package(NumCpp)

target_link_libraries(wav_to_arr 
    PRIVATE
    wavreader
)

target_link_libraries(stft
    PRIVATE
    wavreader
    dsp
    NumCpp::NumCpp
)

# find_package(NumCpp)
# target_link_libraries(stft
#     NumCpp::NumCpp
# )