add_library(quantum_state 
    QuantumStates.cpp
    Statevector.cpp
    DensityMatrix.cpp
    UnitaryState.cpp
    MatrixProductState.cpp
)

set_target_properties(quantum_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(quantum_state PRIVATE quantum_circuit ${ITENSOR_DEPS} itensor fmt::fmt)
target_include_directories(quantum_state PUBLIC ${ITENSOR_DIR})
target_link_directories(quantum_state PUBLIC ${ITENSOR_DIR}/lib ${BLAS_DIR} ${LAPACK_DIR})
