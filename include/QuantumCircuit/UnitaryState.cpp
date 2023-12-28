#include "QuantumStates.h"

UnitaryState::UnitaryState(uint32_t num_qubits) : QuantumState(num_qubits) {
  unitary = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
  unitary.setIdentity();
}

std::string UnitaryState::to_string() const {
  return get_statevector().to_string();
}

void UnitaryState::evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) {
  Eigen::MatrixXcd full_gate = full_circuit_unitary(gate, qubits, num_qubits);
  evolve(full_gate);
}

void UnitaryState::evolve(const Eigen::MatrixXcd &gate) {
  unitary = gate * unitary;
}

void UnitaryState::normalize() {
  unitary = normalize_unitary(unitary);
}

Statevector UnitaryState::get_statevector() const {
  Statevector statevector(num_qubits);
  statevector.evolve(unitary);
  return statevector;
}

double UnitaryState::entropy(const std::vector<uint32_t> &sites, uint32_t index) {
  return get_statevector().entropy(sites, index);
}
