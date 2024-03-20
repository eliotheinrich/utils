add_library(pyqtools_misc
    BinaryMatrix.hpp
    BinaryPolynomial.hpp
)
set_target_properties(pyqtools_misc PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
