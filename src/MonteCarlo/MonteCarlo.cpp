#include "MonteCarlo.hpp"

#include <fmt/format.h>

#ifdef BUILD_GLFW
#include <GLFW/glfw3.h>
#endif

void MonteCarloSimulator::key_callback(int key) {
#ifdef BUILD_GLFW
  if (key == GLFW_KEY_DOWN) {
    std::cout << fmt::format("Lowering temperature: T = {:.2f}\n", temperature);
    temperature = std::max(0.0, temperature - 0.1);
  } else if (key == GLFW_KEY_UP) {
    std::cout << fmt::format("Raising temperature: T = {:.2f}\n", temperature);
    temperature = temperature + 0.1;
  }
#endif
}

MonteCarloSimulator::~MonteCarloSimulator()=default;
