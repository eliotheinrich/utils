add_library(samplers 
    InterfaceSampler.hpp
    InterfaceSampler.cpp
	EntropySampler.hpp
	QuantumStateSampler.hpp
)

set_target_properties(samplers PROPERTIES 
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(samplers PRIVATE ${PFFFT_LIB})
target_include_directories(samplers PRIVATE ${QUANTUMSTATE_DIR} ${PFFFT_DIR})

