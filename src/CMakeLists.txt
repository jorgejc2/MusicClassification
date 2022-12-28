project(MusicClassification)

add_library(wavreader wavread.cpp)
add_library(dsp dsp.cpp)

target_include_directories(wavreader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(dsp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)