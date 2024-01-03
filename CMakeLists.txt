cmake_minimum_required(VERSION 3.27)

project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------- #
# -- Manually installed libraries -- #
# ---------------------------------- #

set(PFFFT_DIR $ENV{PFFFT_DIR})
find_library(PFFFT_LIB pffft HINTS ${PFFFT_DIR}/lib)
add_compile_definitions(PFFFT_ENABLE_DOUBLE)

set(BLAS_DIR $ENV{BLAS_PATH})
set(LAPACK_DIR $ENV{LAPACK_PATH})
set(LAPACKBLAS gfortran lapack openblas)
set(LAPACKBLAS_DIR ${LAPACK_DIR} ${BLAS_DIR})

set(ITENSOR_DIR $ENV{ITENSOR_DIR})
add_library(itensor STATIC IMPORTED)
set_target_properties(itensor PROPERTIES IMPORTED_LOCATION ${ITENSOR_DIR}/lib/libitensor.a)

# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

include(FetchContent)

set(EXTERNALS "")

if (DEFINED DATAFRAME_SOURCE)
    set(dataframe_SOURCE_DIR ${DATAFRAME_SOURCE})
else()
    FetchContent_Declare(
        dataframe
        GIT_REPOSITORY https://github.com/eliotheinrich/dataframe
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    list(APPEND EXTERNALS dataframe)
endif()

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
list(APPEND EXTERNALS eigen)

FetchContent_MakeAvailable(${EXTERNALS})

include_directories(
    ${dataframe_SOURCE_DIR}/src
    ${eigen_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

add_library(graph
    include/Graph.hpp
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set_target_properties(graph PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)

set(CLIFFORD_DIR ${PROJECT_SOURCE_DIR}/include/CliffordState)
set(QUANTUMSTATE_DIR ${PROJECT_SOURCE_DIR}/include/QuantumState)
set(SAMPLERS_DIR ${PROJECT_SOURCE_DIR}/include/Samplers)
set(MONTECARLO_DIR ${PROJECT_SOURCE_DIR}/include/MonteCarlo)

add_subdirectory(${CLIFFORD_DIR})
add_subdirectory(${SAMPLERS_DIR})
add_subdirectory(${QUANTUMSTATE_DIR})
add_subdirectory(${MONTECARLO_DIR})
