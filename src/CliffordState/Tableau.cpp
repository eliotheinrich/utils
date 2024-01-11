#include "Tableau.hpp"

tableau_utils::Circuit PauliString::reduce(bool z = true) const {
  Tableau tableau = Tableau(num_qubits, std::vector<PauliString>{*this});

  tableau_utils::Circuit circuit;

  if (z) {
    tableau.h(0);
    circuit.push_back(tableau_utils::hgate{0});
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    if (tableau.z(0, i)) {
      if (tableau.x(0, i)) {
        tableau.s(i);
        circuit.push_back(tableau_utils::sgate{i});
      } else {
        tableau.h(i);
        circuit.push_back(tableau_utils::hgate{i});
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
      circuit.push_back(tableau_utils::cxgate{q1, q2});
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

        circuit.push_back(tableau_utils::cxgate{0, ql});
        circuit.push_back(tableau_utils::cxgate{ql, 0});
        circuit.push_back(tableau_utils::cxgate{0, ql});

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
    circuit.push_back(tableau_utils::sgate{0});
    circuit.push_back(tableau_utils::sgate{0});
    circuit.push_back(tableau_utils::hgate{0});
    circuit.push_back(tableau_utils::sgate{0});
    circuit.push_back(tableau_utils::sgate{0});
    circuit.push_back(tableau_utils::hgate{0});
  }

  if (z) {
    // tableau is discarded after function exits, so no need to apply it here. Just add to circuit.
    circuit.push_back(tableau_utils::hgate{0});
  }

  return circuit;
}

tableau_utils::Circuit PauliString::transform(PauliString const &p) const {
  tableau_utils::Circuit c1 = reduce();
  tableau_utils::Circuit c2 = conjugate_circuit(p.reduce());

  c1.insert(c1.end(), c2.begin(), c2.end());

  return c1;
}
