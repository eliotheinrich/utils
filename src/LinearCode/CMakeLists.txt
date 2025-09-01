add_library(linear_code
    LinearCode.cpp
)

target_include_directories(linear_code PUBLIC ${qutils_SOURCE_DIR}/src)
