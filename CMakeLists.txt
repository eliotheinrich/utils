cmake_minimum_required(VERSION 3.26)
project(utils LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -w")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g") 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

find_package(dataframe REQUIRED)
find_package(qutils REQUIRED)
find_package(fmt REQUIRED)

# -------------------------------- #
# -------- Adding code ----------- #
# -------------------------------- #

include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

if (DEFINED BUILD_GLFW)
  add_compile_definitions(BUILD_GLFW)
  set(DISPLAY_DIR ${PROJECT_SOURCE_DIR}/src/SimulatorDisplay)
  add_subdirectory(${DISPLAY_DIR})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/src/LinearCode)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/MonteCarlo)

if (BUILDING_PYUTILS)
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
    utils_bindings
    STABLE_ABI
    NB_STATIC
    src/PyUtils.hpp
    src/PyUtils.cpp
  )

  set(utils_libs dataframe::dataframe linear_code qutils::clifford_state fmt::fmt)
  if (DEFINED BUILD_GLFW)
    list(APPEND utils_libs simulator_display)
  endif()

  target_link_libraries(utils_bindings PUBLIC ${utils_libs})

  install(
    TARGETS utils_bindings
    LIBRARY DESTINATION utils # TODO rename something less generic
  )
endif()
