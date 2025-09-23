add_library(montecarlo MonteCarlo.cpp)
target_include_directories(montecarlo PUBLIC ${qutils_SOURCE_DIR}/src)
target_link_libraries(montecarlo PUBLIC fmt::fmt dataframe::dataframe)
