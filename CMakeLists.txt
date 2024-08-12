cmake_minimum_required(VERSION 3.27)

project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g") 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------- #
# -- Manually installed libraries -- #
# ---------------------------------- #

set(PFFFT_DIR $ENV{PFFFT_DIR})
find_library(PFFFT_LIB pffft HINTS ${PFFFT_DIR}/lib)
add_compile_definitions(PFFFT_ENABLE_DOUBLE)

find_package(fmt)

set(BLAS_DIR $ENV{BLAS_DIR})
set(LAPACK_DIR $ENV{LAPACK_DIR})
set(LAPACKBLAS lapack openblas)
set(LAPACKBLAS_DIR ${LAPACK_DIR} ${BLAS_DIR})

set(ITENSOR_DIR $ENV{ITENSOR_DIR})

set(EXTERNAL_DIRS ${LAPACK_DIR} ${BLAS_DIR} ${ITENSOR_DIR}/lib)

# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

include(FetchContent)

set(EXTERNALS "")

if (DEFINED DATAFRAME_SOURCE)
    set(dataframe_SOURCE_DIR ${DATAFRAME_SOURCE})
    add_subdirectory(${DATAFRAME_SOURCE} "dataframe_src")
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
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG main
)
list(APPEND EXTERNALS glaze)

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
    ${PROJECT_SOURCE_DIR}/src
    ${glaze_SOURCE_DIR}/include
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CLIFFORD_DIR ${PROJECT_SOURCE_DIR}/src/CliffordState)
set(QUANTUMSTATE_DIR ${PROJECT_SOURCE_DIR}/src/QuantumState)
set(SAMPLERS_DIR ${PROJECT_SOURCE_DIR}/src/Samplers)
set(MONTECARLO_DIR ${PROJECT_SOURCE_DIR}/src/MonteCarlo)
set(MISC_DIR ${PROJECT_SOURCE_DIR}/src/Misc)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/src/Tests)
if (DEFINED BUILD_GLFW)
    add_compile_definitions(BUILD_GLFW)
    set(DISPLAY_DIR ${PROJECT_SOURCE_DIR}/src/SimulatorDisplay)
    add_subdirectory(${DISPLAY_DIR})
endif()

add_subdirectory(${QUANTUMSTATE_DIR})
add_subdirectory(${CLIFFORD_DIR})
add_subdirectory(${MONTECARLO_DIR})
add_subdirectory(${MISC_DIR})
add_subdirectory(${SAMPLERS_DIR})
add_subdirectory(${TEST_DIR})

if (BUILDING_PYQTOOLS)
    find_package(Python 3.10 
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule
    )

    # Finding nanobind
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
    )
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        pyqtools_bindings
        STABLE_ABI
        NB_STATIC
        src/PyUtils.cpp
    )

    target_link_libraries(pyqtools_bindings PUBLIC ${LAPACKBLAS} itensor)
    target_include_directories(pyqtools_bindings PUBLIC ${ITENSOR_DIR})
    target_link_directories(pyqtools_bindings PUBLIC ${ITENSOR_DIR}/lib ${BLAS_DIR} ${LAPACK_DIR})
    target_link_libraries(pyqtools_bindings PRIVATE quantum_state clifford_state linear_code ${LAPACKBLAS} itensor)

    install(
        TARGETS pyqtools_bindings
        LIBRARY DESTINATION pyqtools
    )
endif()
