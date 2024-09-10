if (DEFINED PYQTOOLS_BUILDING_TEST)
    add_executable(pyqtools_test
        test.cpp
    )

    set_target_properties(pyqtools_test PROPERTIES 
        LINKER_LANGUAGE CXX 
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(pyqtools_test PRIVATE quantum_state clifford_state linear_code gfortran ${LAPACKBLAS} itensor fmt)
    target_link_directories(pyqtools_test PUBLIC "/home/heinriea/.local/lapack/lib64" "/home/heinriea/.local/openblas/lib64" "/home/heinriea/Builds/itensor/lib" "/home/heinriea/.local/fmt/lib64")
    target_include_directories(pyqtools_test PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR} ${GLAD_PATH}/include)
endif()
