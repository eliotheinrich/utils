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

find_library(ITENSOR
    NAMES itensor
    HINTS "$ENV{ITENSOR_DIR}/lib"
)

set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG REQUIRED)
if (${MKL_FOUND})
    set(ITENSOR_DEPS MKL::MKL ${ITENSOR})
else()
    find_package(OpenBLAS CONFIG)
    set(ITENSOR_DEPS ${OpenBLAS_LIBRARIES} ${ITENSOR})
endif()

if (gfortran)
    set(ITENSOR_DEPS gfortran ${ITENSOR_DEPS})
endif()

target_link_libraries(quantum_state PRIVATE quantum_circuit ${ITENSOR_DEPS})
target_include_directories(quantum_state PRIVATE $ENV{ITENSOR_DIR})
