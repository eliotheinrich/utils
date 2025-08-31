add_library(montecarlo MonteCarlo.cpp)
target_include_directories(montecarlo PUBLIC ${qutils_SOURCE_DIR}/src)
