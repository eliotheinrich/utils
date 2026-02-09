set(utils_test_libs fmt::fmt qutils::qutils linear_code)

foreach(tgt IN ITEMS test_linear_code)
  add_executable(${tgt} ${tgt}.cpp)
  target_link_libraries(${tgt} PRIVATE ${utils_test_libs})
endforeach()
