add_library(clifford_state 
	CliffordState.hpp
	Tableau.cpp
	QuantumCHPState.hpp
	QuantumGraphState.cpp
)
target_link_libraries(clifford_state PRIVATE samplers quantum_state)
set_target_properties(clifford_state PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(clifford_state PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR})
