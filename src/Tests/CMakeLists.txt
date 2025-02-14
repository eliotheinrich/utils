if (DEFINED QUTILS_BUILDING_TEST)
  set(qutils_test_libs quantum_state clifford_state linear_code ${LAPACKBLAS} itensor fmt::fmt)
  if (DEFINED BUILD_GLFW)
    list(APPEND qutils_test_libs simulator_display)
  endif()

  add_executable(qutils_test
    test.cpp
  )

  set_target_properties(qutils_test PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
  )
  
  target_link_libraries(qutils_test PRIVATE ${qutils_test_libs})
  target_include_directories(qutils_test PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR} ${GLAD_PATH}/include)

  add_executable(qutils_benchmark
    benchmark.cpp
  )

  set_target_properties(qutils_test PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
  )

  target_link_libraries(qutils_benchmark PRIVATE ${qutils_test_libs})
  target_include_directories(qutils_benchmark PRIVATE ${SAMPLERS_DIR} ${QUANTUMSTATE_DIR} ${CLIFFORDSTATE_DIR} ${MONTECARLO_DIR} ${MISC_DIR} ${GLAD_PATH}/include)
endif()
