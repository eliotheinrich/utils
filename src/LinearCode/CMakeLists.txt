add_library(linear_code
    LinearCode.cpp
)

set_target_properties(linear_code PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
