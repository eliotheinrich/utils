#include "CircuitUtils.h"

#include <iostream>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <unordered_set>
#include <assert.h>


bool qargs_unique(const std::vector<uint32_t>& qargs) {
  std::unordered_set<uint32_t> unique;
  for (auto const &q : qargs) {
    if (unique.count(q) > 0) {
      return false;
    }
    unique.insert(q);
  }

  return true;
}

std::vector<uint32_t> parse_qargs_opt(const std::optional<std::vector<uint32_t>>& qargs_opt, uint32_t num_qubits) {
  std::vector<uint32_t> qargs;
  if (qargs_opt.has_value()) {
    qargs = qargs_opt.value();
    for (uint32_t i = 0; i < qargs.size(); i++) {
      assert(qargs[i] >= 0 && qargs[i] < num_qubits);
    }
  } else {
    qargs = std::vector<uint32_t>(num_qubits);
    std::iota(qargs.begin(), qargs.end(), 0);
  }

  return qargs;
}

std::vector<uint32_t> complement(const std::vector<uint32_t>& sites, size_t num_qubits) {
  std::vector<uint32_t> sites_(sites.size());
  for (size_t i = 0; i < sites.size(); i++) {
    sites_[i] = num_qubits - sites[i] - 1;
  }

  return sites_;
}

std::pair<uint32_t, uint32_t> get_targets(uint32_t d, uint32_t q, uint32_t num_qubits) {
  if (d % 2 == 0) {
    uint32_t q1 = (2*q) % num_qubits;
    uint32_t q2 = (2*q + 1) % num_qubits;
    return std::pair<uint32_t, uint32_t>(q1, q2);
  } else {
    uint32_t q1 = (2*q + 1) % num_qubits;
    uint32_t q2 = (2*q + 2) % num_qubits;
    return std::pair<uint32_t, uint32_t>(q1, q2);

  }
}

Eigen::MatrixXcd haar_unitary(uint32_t num_qubits, std::minstd_rand &rng) {
  Eigen::MatrixXcd z = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (uint32_t r = 0; r < z.rows(); r++) {
    for (uint32_t c = 0; c < z.cols(); c++) {
      z(r, c) = std::complex<double>(distribution(rng), distribution(rng));
    }
  }

  Eigen::MatrixXcd q, r;
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(z);

  q = qr.householderQ();
  r = qr.matrixQR().triangularView<Eigen::Upper>();

  Eigen::MatrixXcd d = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
  d.diagonal() = r.diagonal().cwiseQuotient(r.diagonal().cwiseAbs());

  return q * d;
}

Eigen::MatrixXcd haar_unitary(uint32_t num_qubits) {
  thread_local std::random_device rng;
  std::minstd_rand gen(rng());
  return haar_unitary(num_qubits, gen);
}

Eigen::MatrixXcd random_real_unitary(std::minstd_rand &rng) {
  Eigen::MatrixXcd u = Eigen::MatrixXcd(4u, 4u);

  double t1 = double(rng())/double(RAND_MAX);
  double t2 = double(rng())/double(RAND_MAX);
  double ct1 = std::cos(t1);
  double ct2 = std::cos(t2);
  double st1 = std::sin(t1);
  double st2 = std::sin(t2);

  u << ct1*ct2, ct1*st2, ct2*st1,-st1*st2,
    -ct1*st2, ct1*ct2,-st1*st2,-ct2*st1,
    -ct2*st1,-st1*st2, ct1*ct2,-ct1*st2,
    st1*st2,-ct2*st1,-ct1*st2,-ct1*ct2;

  return u;
}

Eigen::MatrixXcd random_real_unitary() {
  thread_local std::random_device rng;
  std::minstd_rand gen(rng());
  return random_real_unitary(gen);
}

Eigen::MatrixXcd full_circuit_unitary(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits, uint32_t total_qubits) {
  if (total_qubits < qubits.size()) {
    throw std::invalid_argument("Too many qubits provided for gate.");
  }

  if (!((1u << qubits.size()) == gate.rows() && (1u << qubits.size()) == gate.cols())) {
    throw std::invalid_argument("Gate has invalid dimensions for provided qubits.");
  }

  uint32_t s = 1u << total_qubits;
  uint32_t h = 1u << qubits.size();

  Eigen::MatrixXcd full_gate = Eigen::MatrixXcd::Zero(s, s);
  for (uint32_t i = 0; i < s; i++) {
    uint32_t r = 0;
    for (uint32_t k = 0; k < qubits.size(); k++) {
      uint32_t x = (i >> qubits[k]) & 1u;
      uint32_t p = k;
      r = (r & ~(1u << p)) | (x << p);
    }

    for (uint32_t c = 0; c < h; c++) {
      uint32_t j = i;
      // j is total bits
      // c is subsystem bit
      // set the q[k]th bit of j equal to the kth bit of c
      for (uint32_t k = 0; k < qubits.size(); k++) {
        uint32_t p = k;
        uint32_t x = (c >> p) & 1u;
        j = (j & ~(1u << qubits[k])) | (x << qubits[k]);
      }

      full_gate(i, j) = gate(r, c);
    }
  }

  return full_gate;
}

Eigen::MatrixXcd normalize_unitary(Eigen::MatrixXcd &unitary) {
  auto QR = unitary.householderQr();
  Eigen::MatrixXcd Q = QR.householderQ();

  return Q;
}
