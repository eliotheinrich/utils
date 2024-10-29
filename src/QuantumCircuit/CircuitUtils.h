#pragma once

#include <vector>
#include <random>
#include <optional>
#include <utility>
#include <Eigen/Dense>

namespace quantumcircuit_utils {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

bool qargs_unique(const std::vector<uint32_t>& qargs);
std::vector<uint32_t> parse_qargs_opt(const std::optional<std::vector<uint32_t>>& qargs_opt, uint32_t num_qubits);
std::pair<uint32_t, uint32_t> get_targets(uint32_t d, uint32_t q, uint32_t num_qubits);

Eigen::MatrixXcd haar_unitary(uint32_t num_qubits, std::minstd_rand &rng);
Eigen::MatrixXcd haar_unitary(uint32_t num_qubits);

Eigen::MatrixXcd random_real_unitary(std::minstd_rand &rng);
Eigen::MatrixXcd random_real_unitary();

Eigen::MatrixXcd full_circuit_unitary(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits, uint32_t total_qubits);

Eigen::MatrixXcd normalize_unitary(Eigen::MatrixXcd &unitary);
