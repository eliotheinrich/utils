#include "Instructions.hpp"
#include "PauliString.hpp"

const std::unordered_map<SymbolicGate::GateLabel, Eigen::MatrixXcd> SymbolicGate::gate_map = {
  { SymbolicGate::GateLabel::H,      process_gate_data(gates::H::value)},
  { SymbolicGate::GateLabel::X,      process_gate_data(gates::X::value)},
  { SymbolicGate::GateLabel::Y,      process_gate_data(gates::Y::value)},
  { SymbolicGate::GateLabel::Z,      process_gate_data(gates::Z::value)},
  { SymbolicGate::GateLabel::sqrtX,  process_gate_data(gates::sqrtX::value)},
  { SymbolicGate::GateLabel::sqrtY,  process_gate_data(gates::sqrtY::value)},
  { SymbolicGate::GateLabel::S,      process_gate_data(gates::sqrtZ::value)},
  { SymbolicGate::GateLabel::sqrtXd, process_gate_data(gates::sqrtXd::value)},
  { SymbolicGate::GateLabel::sqrtYd, process_gate_data(gates::sqrtYd::value)},
  { SymbolicGate::GateLabel::Sd,     process_gate_data(gates::sqrtZd::value)},
  { SymbolicGate::GateLabel::T,      process_gate_data(gates::T::value)},
  { SymbolicGate::GateLabel::Td,     process_gate_data(gates::Td::value)},
  { SymbolicGate::GateLabel::CX,     process_gate_data(gates::CX::value)},
  { SymbolicGate::GateLabel::CY,     process_gate_data(gates::CY::value)},
  { SymbolicGate::GateLabel::CZ,     process_gate_data(gates::CZ::value)},
  { SymbolicGate::GateLabel::SWAP,   process_gate_data(gates::SWAP::value)},
};

Measurement::Measurement(const Qubits& qubits, std::optional<PauliString> pauli, std::optional<bool> outcome)
: qubits(qubits), pauli(pauli), outcome(outcome) {
  PauliString p = pauli ? pauli.value() : PauliString("+Z");
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

Measurement Measurement::computational_basis(uint32_t q, std::optional<bool> outcome) {
  return Measurement({q}, PauliString("+Z"), outcome);
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

Qubits get_instruction_support(const Instruction& inst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const std::shared_ptr<Gate> gate) { 
      return gate->qubits;
    },
    [](const Measurement& m) { 
      return m.qubits;
    },
    [](const WeakMeasurement& m) {
      return m.qubits;
    }
  }, inst);
}

bool instruction_is_unitary(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](std::shared_ptr<Gate> gate) { return true; },
    [](const Measurement& m) { return false; },
    [](const WeakMeasurement& m) { return false; }
  }, inst);
}
