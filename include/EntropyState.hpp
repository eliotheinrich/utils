#pragma once

#include <vector>
#include <cstdint>
#include <numeric>

class EntropyState {
  protected:
    uint32_t system_size;

  public:
    EntropyState()=default;

    EntropyState(uint32_t system_size) : system_size(system_size) {}

    virtual double entropy(const std::vector<uint32_t>& sites, uint32_t index)=0;

    template <typename T = double>
    T cum_entropy(uint32_t i, uint32_t index = 2u, bool direction = true) {
      if (direction) { // Left-oriented cumulative entropy
        std::vector<uint32_t> sites(i+1);
        std::iota(sites.begin(), sites.end(), 0);
        return static_cast<T>(entropy(sites, index));
      } else { // Right-oriented cumulative entropy
        std::vector<uint32_t> sites(system_size - i);
        std::iota(sites.begin(), sites.end(), i);
        return static_cast<T>(entropy(sites, index));
      }
    }

    template <typename T = double>
    std::vector<T> get_entropy_surface(uint32_t index=2u) {
      std::vector<T> entropy_surface(system_size);

      for (uint32_t i = 0; i < system_size; i++) {
        entropy_surface[i] = cum_entropy<T>(i, index);
      }

      return entropy_surface;
    }
};

// TODO find a better place for this to live
static inline uint32_t mod(int a, int b) {
  int c = a % b;
  return (c < 0) ? c + b : c;
}
