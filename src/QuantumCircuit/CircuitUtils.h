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

using QubitInterval = std::optional<std::pair<uint32_t, uint32_t>>;
using Qubits = std::vector<uint32_t>;
using QubitSupport = std::variant<Qubits, QubitInterval>;

namespace quantumcircuit_utils {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

static QubitInterval to_interval(const QubitSupport& support) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](const QubitInterval& interval) -> QubitInterval {
      return interval;
    },
    [](const Qubits& qubits) -> QubitInterval {
      if (qubits.size() == 0) {
        return std::nullopt;
      }
      Qubits sorted(qubits.begin(), qubits.end());
      for (size_t i = 0; i < sorted.size() - 1; i++) {
        if (sorted[i + 1] != sorted[i] + 1) {
          throw std::runtime_error(fmt::format("Qubits {} are not an interval.", qubits));
        }
      }

      return std::make_pair(sorted[0], sorted[sorted.size() - 1] + 1);
    }
  }, support);
}

static Qubits to_qubits(const QubitSupport& support) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](const QubitInterval& interval) -> Qubits {
      if (interval) {
        auto interval_ = interval.value();
        uint32_t q1 = std::min(interval_.first, interval_.second);
        uint32_t q2 = std::max(interval_.first, interval_.second);
        size_t num_qubits = q2 - q1;
        Qubits qubits(num_qubits);
        std::iota(qubits.begin(), qubits.end(), q1);
        return qubits;
      } else {
        return {};
      }
    },
    [](const Qubits& qubits) -> Qubits {
      return qubits;
    }
  }, support);
}

static QubitInterval support_range(const QubitSupport& support) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](const QubitInterval& interval) -> QubitInterval {
      return interval;
    },
    [](const Qubits& qubits) -> QubitInterval {
      if (qubits.size() == 0) {
        return std::nullopt;
      }
      auto [min, max] = std::ranges::minmax(qubits);
      return std::make_pair(min, max + 1);
    }
  }, support);
}

static QubitSupport support_complement(const QubitSupport& support, size_t n) {
  auto interval = support_range(support);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > n) {
      throw std::runtime_error(fmt::format("Support on [{}, {}) cannot be complemented on {} qubits.", q1, q2, n));
    }
  }

  std::vector<bool> mask(n, true);
  auto qubits = to_qubits(support);
  for (const auto q : qubits) {
    mask[q] = false;
  }

  Qubits qubits_;
  for (size_t i = 0; i < n; i++) {
    if (mask[i]) {
      qubits_.push_back(i);
    }
  }

  return QubitSupport{qubits_};
}

static bool support_contiguous(const QubitSupport& support) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](const QubitInterval& interval) -> bool{
      return true;
    },
    [](const Qubits& qubits) -> bool {
      Qubits qubits_sorted = qubits;
      std::sort(qubits_sorted.begin(), qubits_sorted.end());

      for (size_t i = 0; i < qubits_sorted.size() - 1; i++) {
        if (qubits_sorted[i+1] != qubits_sorted[i] + 1) {
          return false;
        }
      }

      return true;
    }
  }, support);
}



bool qargs_unique(const Qubits& qubits);
Qubits parse_qargs_opt(const std::optional<Qubits>& qubits_opt, uint32_t num_qubits);
std::pair<uint32_t, uint32_t> get_targets(uint32_t d, uint32_t q, uint32_t num_qubits);
Qubits complement(const Qubits& qubits, size_t num_qubits);


Eigen::MatrixXcd haar_unitary(uint32_t num_qubits);

Eigen::MatrixXcd random_real_unitary();

Eigen::MatrixXcd full_circuit_unitary(const Eigen::MatrixXcd &gate, const Qubits &qubits, uint32_t total_qubits);

Eigen::MatrixXcd normalize_unitary(Eigen::MatrixXcd &unitary);
