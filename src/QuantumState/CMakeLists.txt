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
set_target_properties(quantum_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(quantum_state PRIVATE ${ITENSOR_DIR})
