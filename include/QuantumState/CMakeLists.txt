add_library(quantum_state 
    CircuitUtils.cpp
    Instructions.hpp
    Gates.hpp
    QuantumCircuit.cpp
    Statevector.cpp
    DensityMatrix.cpp
    UnitaryState.cpp
    MatrixProductState.cpp
)

target_link_libraries(quantum_state PRIVATE ${LAPACKBLAS} itensor)
target_link_directories(quantum_state PUBLIC ${LAPACK_DIR} ${BLAS_DIR})
target_include_directories(quantum_state PRIVATE ${ITENSOR_DIR} ${SAMPLERS_DIR})
