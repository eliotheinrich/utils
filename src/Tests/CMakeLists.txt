add_executable(pyqtools_test
    test.cpp
)

set_target_properties(pyqtools_test PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(pyqtools_test PRIVATE quantum_state clifford_state linear_code)
target_include_directories(pyqtools_test PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR})
