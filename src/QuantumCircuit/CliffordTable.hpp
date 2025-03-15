#pragma once

#include "PauliString.hpp"
#include "QuantumCircuit.h"

class CliffordTable {
  private:
    using TableauBasis = std::tuple<PauliString, PauliString, size_t>;
    std::vector<TableauBasis> circuits;

  public:
    CliffordTable()=default;

    CliffordTable(std::function<bool(const QuantumCircuit&)> filter) : circuits() {
      std::vector<std::pair<PauliString, PauliString>> basis;

      for (size_t s1 = 1; s1 < 16; s1++) {
        PauliString X, Z;
        X = PauliString::from_bitstring(2, s1);
        for (size_t s2 = 1; s2 < 16; s2++) {
          Z = PauliString::from_bitstring(2, s2);
          
          // Z should anticommute with X
          if (Z.commutes(X)) {
            continue;
          }

          basis.push_back({ X,  Z });
          basis.push_back({ X, -Z });
          basis.push_back({-X,  Z });
          basis.push_back({-X, -Z });
        }
      }

      Qubits qubits{0, 1};
      for (const auto& [X, Z] : basis) {
        for (size_t r = 0; r < 24; r++) {
          QuantumCircuit qc(2);
          reduce_paulis(X, Z, qubits, qc);
          single_qubit_clifford_impl(qc, 0, r);
          if (filter(qc)) {
            circuits.push_back(std::make_tuple(X, Z, r));
          }
        }
      }
    }

    CliffordTable(const CliffordTable& other) : circuits(other.circuits) {}

    size_t num_elements() const {
      return circuits.size();
    }

    QuantumCircuit get_circuit(uint32_t r) const {
      QuantumCircuit qc(2);
      auto [X, Z, r2] = circuits[r];

      reduce_paulis(X, Z, {0, 1}, qc);
      single_qubit_clifford_impl(qc, 0, r2);

      return qc;
    }

    std::vector<QuantumCircuit> get_circuits() const {
      std::vector<QuantumCircuit> circuits(num_elements());
      size_t r = 0;
      std::generate(circuits.begin(), circuits.end(), [&r, this]() { return get_circuit(r++); });
      return circuits;
    }

    template <typename... Args>
    void apply_random(const Qubits& qubits, Args&... args) {
      size_t r1 = randi() % circuits.size();

      auto [X, Z, r2] = circuits[r1];

      reduce_paulis(X, Z, qubits, args...);
      (single_qubit_clifford_impl(args, qubits[0], r2), ...);
    } 
};

