#pragma once

#include <vector>
#include <cstdint>
#include <numeric>

#include "Support.hpp"

class EntanglementEntropyState {
  protected:
    uint32_t system_size;

  public:
    EntanglementEntropyState()=default;

    EntanglementEntropyState(uint32_t system_size) : system_size(system_size) {}

    virtual double entanglement(const QubitSupport& sites, uint32_t index)=0;

    template <typename T = double>
    T cum_entanglement(uint32_t i, uint32_t index = 2u, bool direction = true) {
      if (direction) { // Left-oriented cumulative entanglement 
        QubitInterval support = std::make_pair(0, i+1);
        return static_cast<T>(entanglement(support, index));
      } else { // Right-oriented cumulative entanglement 
        QubitInterval support = std::make_pair(i, system_size);
        return static_cast<T>(entanglement(support, index));
      }
    }

    template <typename T = double>
    std::vector<T> get_entanglement(uint32_t index=2u) {
      std::vector<T> entanglement(system_size);

      for (uint32_t i = 0; i < system_size; i++) {
        entanglement[i] = cum_entanglement<T>(i, index, true);
      }

      return entanglement;
    }
};
