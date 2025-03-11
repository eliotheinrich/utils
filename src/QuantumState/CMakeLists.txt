add_library(quantum_state 
    QuantumStates.cpp
    Statevector.cpp
    DensityMatrix.cpp
    MatrixProductState.cpp
)

set_target_properties(quantum_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)

find_package(OpenBLAS CONFIG REQUIRED)
find_library(ITENSOR NAMES itensor STATIC)

find_library(ITENSOR
    NAMES itensor
    HINTS "$ENV{ITENSOR_DIR}/lib"
)
include_directories("$ENV{ITENSOR_DIR}")

set(ITENSOR_DEPS ${OpenBLAS_LIBRARIES} ${ITENSOR})
if (gfortran)
    set(ITENSOR_DEPS gfortran ${ITENSOR_DEPS})
endif()

target_link_libraries(quantum_state PRIVATE quantum_circuit ${ITENSOR_DEPS})
