add_library(clifford_state 
    CliffordState.hpp
    CliffordState.cpp
    QuantumCHPState.hpp
    QuantumCHPState.cpp
    QuantumGraphState.cpp
    Tableau.hpp
)
set_target_properties(clifford_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(clifford_state PUBLIC quantum_state)
target_include_directories(clifford_state PRIVATE ${QUANTUMSTATE_DIR})
