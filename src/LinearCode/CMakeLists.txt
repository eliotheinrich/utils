add_library(linear_code
    LinearCode.cpp
)

target_include_directories(linear_code PUBLIC ${qutils_SOURCE_DIR}/src)
target_link_libraries(linear_code PUBLIC fmt::fmt dataframe::dataframe qutils::qutils)
