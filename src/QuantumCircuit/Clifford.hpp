#pragma once

#include "PauliString.hpp"
#include "QuantumCircuit.h"

template <class T>
void single_qubit_clifford_impl(T& qobj, size_t q, size_t r) {
  // r == 0 is identity, so do nothing in this case
  // Conjugates are marked as comments next to each case
  if (r == 1) { // 1
    qobj.x(q);
  } else if (r == 2) { // 2
    qobj.y(q);
  } else if (r == 3) { // 3
    qobj.z(q);
  } else if (r == 4) { // 8
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 5) { // 11
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 6) { // 9
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 7) { // 10
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 8) { // 4
    qobj.h(q);
    qobj.s(q);
  } else if (r == 9) { // 6
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 10) { // 7
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 11) { // 5
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 12) { // 12
    qobj.h(q);
  } else if (r == 13) { // 15
    qobj.h(q);
    qobj.x(q);
  } else if (r == 14) { // 14
    qobj.h(q);
    qobj.y(q);
  } else if (r == 15) { // 13
    qobj.h(q);
    qobj.z(q);
  } else if (r == 16) { // 17
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 17) { // 16
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 18) { // 18
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 19) { // 19
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 20) { // 23
    qobj.s(q);
  } else if (r == 21) { // 21
    qobj.s(q);
    qobj.x(q);
  } else if (r == 22) { // 22
    qobj.s(q);
    qobj.y(q);
  } else if (r == 23) { // 20
    qobj.s(q);
    qobj.z(q);
  }
}

template<typename... Args>
void reduce_paulis_inplace(PauliString& p1, PauliString& p2, const Qubits& qubits, Args&... args) {
  size_t num_qubits = p1.num_qubits;
  if (p2.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot reduce tableau for provided PauliStrings {} and {}; mismatched number of qubits.", p1, p2));
  }

  Qubits qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  p1.reduce_inplace(false, std::make_pair(&args, qubits)..., std::make_pair(&p2, qubits_));

  PauliString z1p = PauliString::basis(num_qubits, "Z", 0, 0);
  PauliString z1m = PauliString::basis(num_qubits, "Z", 0, 2);

  if (p2 != z1p && p2 != z1m) {
    p2.reduce_inplace(true, std::make_pair(&args, qubits)..., std::make_pair(&p1, qubits_));
  }

  uint8_t sa = p1.get_r();
  uint8_t sb = p2.get_r();

  auto interpret_sign = [](uint8_t s) {
    if (s == 0) {
      return false;
    } else if (s == 2) {
      return true;
    } else {
      throw std::runtime_error("Anomolous phase detected in reduce.");
    }
  };

  if (interpret_sign(sa)) {
    if (interpret_sign(sb)) {
      // apply y
      (args.y(qubits[0]), ...);
      p1.y(0);
      p2.y(0);
    } else {
      // apply z
      (args.z(qubits[0]), ...);
      p1.z(0);
      p2.z(0);
    }
  } else {
    if (interpret_sign(sb)) {
      // apply x
      (args.x(qubits[0]), ...);
      p1.x(0);
      p2.x(0);
    }
  }
}

template<typename... Args>
void reduce_paulis(const PauliString& p1, const PauliString& p2, const Qubits& qubits, Args&... args) {
  PauliString p1_ = p1;
  PauliString p2_ = p2;

  size_t num_qubits = p1.num_qubits;

  Qubits qubits_ = argsort(qubits);

  QuantumCircuit qc(num_qubits);
  reduce_paulis_inplace(p1_, p2_, qubits_, qc);
  qc.apply(qubits, args...);

 // reduce_paulis_inplace(p1_, p2_, qubits, args...);
}

// Performs an iteration of the random clifford algorithm outlined in https://arxiv.org/pdf/2008.06011.pdf
template <typename... Args>
void random_clifford_iteration_impl(const Qubits& qubits, Args&... args) {
  size_t num_qubits = qubits.size();

  // If only acting on one qubit, can easily lookup from a table
  if (num_qubits == 1) {
    size_t r = randi() % 24;
    (single_qubit_clifford_impl(args, {qubits[0]}, r), ...);
    return;
  }

  Qubits qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  PauliString p1 = PauliString::randh(num_qubits);
  PauliString p2 = PauliString::randh(num_qubits);
  while (p1.commutes(p2)) {
    p2 = PauliString::randh(num_qubits);
  }

  reduce_paulis_inplace(p1, p2, qubits, args...);
}

template <typename... Args>
void random_clifford_impl(const Qubits& qubits, Args&... args) {
  Qubits qubits_(qubits.begin(), qubits.end());

  for (uint32_t i = 0; i < qubits.size(); i++) {
    random_clifford_iteration_impl(qubits_, args...);
    qubits_.pop_back();
  }
}
