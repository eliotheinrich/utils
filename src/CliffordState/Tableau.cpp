#include "Tableau.hpp"

QuantumCircuit PauliString::reduce(bool z = true) const {
  Tableau tableau = Tableau(num_qubits, std::vector<PauliString>{*this});

  QuantumCircuit circuit;

  if (z) {
    tableau.h(0);
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("h", {0})));
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    if (tableau.z(0, i)) {
      if (tableau.x(0, i)) {
        tableau.s(i);
        circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("s", {i})));
      } else {
        tableau.h(i);
        circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("h", {i})));
      }
    }
  }

  // Step two
  std::vector<uint32_t> nonzero_idx;
  for (uint32_t i = 0; i < num_qubits; i++) {
    if (tableau.x(0, i)) {
      nonzero_idx.push_back(i);
    }
  }
  while (nonzero_idx.size() > 1) {
    for (uint32_t j = 0; j < nonzero_idx.size()/2; j++) {
      uint32_t q1 = nonzero_idx[2*j];
      uint32_t q2 = nonzero_idx[2*j+1];
      tableau.cx(q1, q2);
      circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("cx", {q1, q2})));
    }

    remove_even_indices(nonzero_idx);
  }

  // Step three
  uint32_t ql = nonzero_idx[0];
  if (ql != 0) {
    for (uint32_t i = 0; i < num_qubits; i++) {
      if (tableau.x(0, i)) {
        tableau.cx(0, ql);
        tableau.cx(ql, 0);
        tableau.cx(0, ql);

        circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("cx", {0, ql})));
        circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("cx", {ql, 0})));
        circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("cx", {0, ql})));

        break;
      }
    }
  }

  if (tableau.r(0)) {
    // Apply Y gate to tableau
    tableau.h(0);
    tableau.s(0);
    tableau.s(0);
    tableau.h(0);
    tableau.s(0);
    tableau.s(0);

    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("s", {0})));
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("s", {0})));
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("h", {0})));
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("s", {0})));
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("s", {0})));
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("h", {0})));
  }

  if (z) {
    // tableau is discarded after function exits, so no need to apply it here. Just add to circuit.
    circuit.add_gate(std::shared_ptr<Gate>(new SymbolicGate("h", {0})));
  }

  return circuit;
}

QuantumCircuit PauliString::transform(PauliString const &p) const {
  QuantumCircuit c1 = reduce();
  QuantumCircuit c2 = p.reduce().adjoint();

  c1.append(c2);

  return c1;
}

//template <>
//struct glz::meta<PauliString> {
//  static constexpr auto value = glz::object(
//    "num_qubits", &PauliString::num_qubits,
//    "phase", &PauliString::phase,
//    "width", &PauliString::width,
//    "bit_string", &PauliString::bit_string
//  );
//};
//
//template <>
//struct glz::meta<Tableau> {
//  static constexpr auto value = glz::object(
//    "track_destabilizers", &Tableau::track_destabilizers,
//    "num_qubits", &Tableau::num_qubits,
//    "rows", &Tableau::rows
//  );
//};
