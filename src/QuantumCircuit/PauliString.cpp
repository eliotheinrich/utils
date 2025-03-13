#include "PauliString.hpp"
#include "Instructions.hpp"
#include "QuantumCircuit.h"

void PauliString::evolve(const QuantumCircuit& qc) {
  if (!qc.is_clifford()) {
    throw std::runtime_error("Provided circuit is not Clifford.");
  }

  if (qc.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot evolve a Paulistring with {} qubits with a QuantumCircuit with {} qubits.", num_qubits, qc.num_qubits));
  }

  for (auto const &inst : qc.instructions) {
    std::visit(quantumcircuit_utils::overloaded{
      [this](std::shared_ptr<Gate> gate) { 
        std::string name = gate->label();

        if (name == "H") {
          h(gate->qubits[0]);
        } else if (name == "S") {
          s(gate->qubits[0]);
        } else if (name == "Sd") {
          sd(gate->qubits[0]);
        } else if (name == "X") {
          x(gate->qubits[0]);
        } else if (name == "Y") {
          y(gate->qubits[0]);
        } else if (name == "Z") {
          z(gate->qubits[0]);
        } else if (name == "sqrtX") {
          sqrtX(gate->qubits[0]);
        } else if (name == "sqrtY") {
          sqrtY(gate->qubits[0]);
        } else if (name == "sqrtZ") {
          sqrtZ(gate->qubits[0]);
        } else if (name == "sqrtXd") {
          sqrtXd(gate->qubits[0]);
        } else if (name == "sqrtYd") {
          sqrtYd(gate->qubits[0]);
        } else if (name == "sqrtZd") {
          sqrtZd(gate->qubits[0]);
        } else if (name == "CX") {
          cx(gate->qubits[0], gate->qubits[1]);
        } else if (name == "CY") {
          cy(gate->qubits[0], gate->qubits[1]);
        } else if (name == "CZ") {
          cz(gate->qubits[0], gate->qubits[1]);
        } else if (name == "SWAP") {
          swap(gate->qubits[0], gate->qubits[1]);
        } else {
          throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to PauliString.evolve.", name));
        }
      },
      [](Measurement m) { 
        throw std::runtime_error(fmt::format("Cannot do measure on a single PauliString."));
      },
      [](WeakMeasurement m) { 
        throw std::runtime_error(fmt::format("Cannot do weak measurement on a single PauliString."));
      },
    }, inst);
  }
}

//void PauliString::evolve(const Instruction& inst) {
//}

QuantumCircuit PauliString::transform(PauliString const &p) const {
  Qubits qubits(p.num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  QuantumCircuit qc1(p.num_qubits);
  reduce(true, std::make_pair(&qc1, qubits));

  QuantumCircuit qc2(p.num_qubits);
  p.reduce(true, std::make_pair(&qc2, qubits));

  qc1.append(qc2.adjoint());

  return qc1;
}
