add_executable(pyqtools_test
    test.cpp
)
target_link_libraries(pyqtools_test PRIVATE samplers quantum_state clifford_state)
set_target_properties(pyqtools_test PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(pyqtools_test PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR})
