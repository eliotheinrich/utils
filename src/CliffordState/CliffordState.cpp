#include "CliffordState.hpp"

thread_local std::minstd_rand CliffordState::rng{CliffordState::random_seed()};
