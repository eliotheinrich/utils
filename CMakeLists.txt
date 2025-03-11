cmake_minimum_required(VERSION 3.27)

project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g") 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------------------------- #
# -- Manually installed libraries -- #
# ---------------------------------- #

add_definitions(-DFMT_HEADER_ONLY=On)

find_package(fmt REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

link_directories("$ENV{ITENSOR_DIR}/lib")
include_directories("$ENV{ITENSOR_DIR}" "${CMAKE_PREFIX_PATH}/include")

set(ITENSOR_DEPS ${OpenBLAS_LIBRARIES} itensor)
if (gfortran)
    set(ITENSOR_DEPS gfortran $ITENSOR_DEPS)
endif()

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
  GIT_TAG cc4ce0efe5133c23ec55f07d9e753f7dd1abb0b8
)
list(APPEND EXTERNALS glaze)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3.4
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
set(QUANTUMCIRCUIT_DIR ${PROJECT_SOURCE_DIR}/src/QuantumCircuit)
set(SAMPLERS_DIR ${PROJECT_SOURCE_DIR}/src/Samplers)
set(MONTECARLO_DIR ${PROJECT_SOURCE_DIR}/src/MonteCarlo)
set(MISC_DIR ${PROJECT_SOURCE_DIR}/src/Misc)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/src/Tests)
if (DEFINED BUILD_GLFW)
    add_compile_definitions(BUILD_GLFW)
    set(DISPLAY_DIR ${PROJECT_SOURCE_DIR}/src/SimulatorDisplay)
    add_subdirectory(${DISPLAY_DIR})
endif()

add_subdirectory(${QUANTUMCIRCUIT_DIR})
add_subdirectory(${QUANTUMSTATE_DIR})
add_subdirectory(${CLIFFORD_DIR})
add_subdirectory(${MONTECARLO_DIR})
add_subdirectory(${MISC_DIR})
add_subdirectory(${SAMPLERS_DIR})
add_subdirectory(${TEST_DIR})

if (BUILDING_PYQUTILS)
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
        qutils_bindings
        STABLE_ABI
        NB_STATIC
        src/PyQutils.hpp
        src/PyQutils.cpp
    )

    set(qutils_libs quantum_state quantum_circuit clifford_state linear_code ${ITENSOR_DEPS} dataframe)
    if (DEFINED BUILD_GLFW)
      list(APPEND qutils_libs simulator_display)
    endif()

    target_link_libraries(qutils_bindings PUBLIC ${qutils_libs})

    install(
        TARGETS qutils_bindings
        LIBRARY DESTINATION qutils
    )
endif()
