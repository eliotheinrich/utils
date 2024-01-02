#include "InterfaceSampler.hpp"

#include <pffft.hpp>
using fft_plan = pffft::Fft<double>;

std::vector<double> InterfaceSampler::structure_function(const std::vector<int>& surface) const {
  size_t N = surface.size();
  fft_plan fft(N);

  if (!fft.isValid()) {
    std::string error_message = "surface.size() = " + std::to_string(N) + " is not valid for fft.";
    throw std::invalid_argument(error_message);
  }

  auto input = fft.valueVector();
  auto output = fft.spectrumVector();

  double Ns = std::sqrt(N);

  if (transform_fluctuations) {
    double hb = surface_avg(surface);
    for (size_t i = 0; i < N; i++) {
      input[i] = static_cast<double>(surface[i] - hb)/Ns;
    }
  } else {
    for (uint32_t i = 0; i < N; i++) {
      input[i] = static_cast<double>(surface[i])/Ns;
    }
  }

  fft.forward(input, output);


  size_t spectrum_size = fft.getSpectrumSize();

  std::vector<double> sk(spectrum_size+1);
  for (size_t i = 1; i < spectrum_size; i++) {
    sk[i] = std::real(output[i]*std::conj(output[i]));
  }

  sk[0] = std::pow(output[0].real(), 2);
  sk[spectrum_size] = std::pow(output[0].imag(), 2);

  return sk;
}
