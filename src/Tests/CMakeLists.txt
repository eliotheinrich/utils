add_executable(test
    test.cpp
)
target_link_libraries(test PRIVATE graph samplers quantum_state clifford_state montecarlo misc)
set_target_properties(test PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(test PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR})
