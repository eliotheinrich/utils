add_library(montecarlo
	MonteCarlo.hpp
)

set_target_properties(montecarlo PROPERTIES 
    LINKER_LANGUAGE CXX 
    POSITION_INDEPENDENT_CODE ON
)
