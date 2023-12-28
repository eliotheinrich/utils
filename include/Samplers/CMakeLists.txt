add_library(samplers
	InterfaceSampler.hpp
	EntropySampler.hpp
	QuantumStateSampler.hpp
)

set_target_properties(samplers PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(samplers PRIVATE ${QUANTUMSTATE_DIR})
