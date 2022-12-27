project(MusicClassification)

add_executable(wav_to_arr basic_wav_to_array/main.cpp)
add_executable(stft basic_stft/main.cpp)

find_package(NumCpp 2.6.2 REQUIRED)
target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)

target_link_libraries(wav_to_arr 
    PRIVATE
    wavreader
)

target_link_libraries(stft
    PRIVATE
    wavreader
)