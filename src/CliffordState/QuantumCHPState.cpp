#include "QuantumCHPState.hpp"

std::vector<dataframe::byte_t> QuantumCHPState::serialize() const {
  std::vector<dataframe::byte_t> bytes;
  auto write_error = glz::write_beve(*this, bytes);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing QuantumCHPState to binary: \n{}", glz::format_error(write_error, bytes)));
  }
  return bytes;
}

void QuantumCHPState::deserialize(const std::vector<dataframe::byte_t>& bytes) {
  auto parse_error = glz::read_beve(*this, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error reading QuantumCHPState from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
