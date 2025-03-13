#include "Instructions.hpp"
#include "PauliString.hpp"

Measurement::Measurement(const Qubits& qubits, std::optional<PauliString> pauli, std::optional<bool> outcome)
: qubits(qubits), pauli(pauli), outcome(outcome) {
  PauliString p = pauli ? pauli.value() : PauliString("+Z");
  size_t expected_num_qubits = p.num_qubits;
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("Invalid number of qubits {} passed to measurement of pauli {}.", qubits, pauli.value()));
  }

  if (!p.hermitian()) {
    throw std::runtime_error(fmt::format("Cannot perform measurement on non-Hermitian Pauli string {}.", p));
  }

  if (qubits.size() == 0) {
    throw std::runtime_error("Must perform measurement on nonzero qubits.");
  }
}

PauliString Measurement::get_pauli() const {
  if (pauli) {
    return pauli.value();
  } else {
    return PauliString("+Z");
  }
}

bool Measurement::is_basis() const {
  return !pauli || (pauli == PauliString("+Z"));
}

bool Measurement::is_forced() const {
  return bool(outcome);
}

bool Measurement::get_outcome() const {
  // is_forced() MUST be true, otherwise this will throw an exception
  return outcome.value();
}

WeakMeasurement::WeakMeasurement(const Qubits& qubits, double beta, std::optional<PauliString> pauli, std::optional<bool> outcome)
: qubits(qubits), beta(beta), pauli(pauli), outcome(outcome) {
  PauliString p = pauli ? pauli.value() : PauliString("+Z");
  size_t expected_num_qubits = p.num_qubits;
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("Invalid number of qubits {} passed to weak measurement of pauli {}.", qubits, pauli.value()));
  }
}

PauliString WeakMeasurement::get_pauli() const {
  if (pauli) {
    return pauli.value();
  } else {
    return PauliString("+Z");
  }
}

bool WeakMeasurement::is_forced() const {
  return bool(outcome);
}

bool WeakMeasurement::get_outcome() const {
  // is_forced() MUST be true, otherwise this will throw an exception
  return outcome.value();
}
