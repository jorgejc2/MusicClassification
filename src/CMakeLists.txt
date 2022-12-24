project(MusicClassification)

add_library(wavreader wavread.cpp)

target_include_directories(wavreader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)