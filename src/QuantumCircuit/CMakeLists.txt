add_library(quantum_circuit
    CircuitUtils.cpp
    Instructions.hpp
    Gates.hpp
    QuantumCircuit.cpp
)

target_link_libraries(quantum_circuit PUBLIC fmt::fmt)
