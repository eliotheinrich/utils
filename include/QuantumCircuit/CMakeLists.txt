add_library(quantum_circuit STATIC
    CircuitUtils.cpp
    Instructions.hpp
    Gates.hpp
    QuantumCircuit.cpp
    Statevector.cpp
    DensityMatrix.cpp
    UnitaryState.cpp
    MatrixProductState.cpp
)


target_link_libraries(quantum_circuit PUBLIC ${LAPACKBLAS} itensor samplers)
target_include_directories(quantum_circuit PUBLIC ${ITENSOR_DIR} ${SAMPLERS_DIR})
