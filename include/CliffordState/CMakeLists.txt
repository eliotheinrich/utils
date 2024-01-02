add_library(clifford_state
	CliffordState.hpp
	Tableau.cpp
	QuantumCHPState.hpp
	QuantumGraphState.cpp
)
target_link_libraries(clifford_state PRIVATE graph samplers quantum_state)
target_include_directories(clifford_state PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR})
