add_library(misc
    BinaryMatrix.hpp
    BinaryPolynomial.hpp
)
set_target_properties(misc PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
