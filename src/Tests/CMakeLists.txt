if (DEFINED QUTILS_BUILDING_TEST)
    set(qutils_test_libs quantum_state clifford_state linear_code)
    if (DEFINED BUILD_GLFW)
        list(APPEND qutils_test_libs simulator_display)
    endif()

    add_executable(test_quantum_state
        test_quantum_state.cpp
    )

    set_target_properties(test_quantum_state PROPERTIES 
        LINKER_LANGUAGE CXX 
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(test_quantum_state PUBLIC ${qutils_test_libs})

    add_executable(test_graph
        test_graph.cpp
    )

    set_target_properties(test_graph PROPERTIES 
        LINKER_LANGUAGE CXX 
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(test_graph PUBLIC ${qutils_test_libs})

    add_executable(qutils_benchmark
        benchmark.cpp
    )

    set_target_properties(qutils_benchmark PROPERTIES 
        LINKER_LANGUAGE CXX 
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(qutils_benchmark PUBLIC ${qutils_test_libs})
endif()
