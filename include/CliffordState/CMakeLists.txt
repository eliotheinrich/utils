add_library(clifford_state
	CliffordState.hpp
	Tableau.cpp
	QuantumCHPState.hpp
	QuantumGraphState.cpp
)
target_link_libraries(clifford_state graph samplers quantum_circuit)
target_include_directories(clifford_state PUBLIC ${CMAKE_SOURCE_DIR}/src ${SAMPLERS_DIR} ${QUANTUMCIRCUIT_DIR} ${CMAKE_SOURCE_DIR}/include)
