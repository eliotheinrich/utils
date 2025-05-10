#pragma once

#include <vector>
#include <variant>
#include <optional>
#include <utility>
#include <Eigen/Dense>

#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "Random.hpp"
#include "Support.hpp"

namespace quantumcircuit_utils {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

bool qargs_unique(const Qubits& qubits);
Qubits parse_qargs_opt(const std::optional<Qubits>& qubits_opt, uint32_t num_qubits);
std::pair<uint32_t, uint32_t> get_targets(uint32_t d, uint32_t q, uint32_t num_qubits);
Qubits complement(const Qubits& qubits, size_t num_qubits);

Eigen::MatrixXcd haar_unitary(uint32_t num_qubits);

Eigen::MatrixXcd random_real_unitary();

Eigen::MatrixXcd full_circuit_unitary(const Eigen::MatrixXcd &gate, const Qubits &qubits, uint32_t total_qubits);

Eigen::MatrixXcd normalize_unitary(Eigen::MatrixXcd &unitary);
