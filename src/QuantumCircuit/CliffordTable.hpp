#pragma once

#include "Clifford.hpp"

template <typename CircuitType = QuantumCircuit>
class CliffordTable {
  private:
    using TableauBasis = std::tuple<PauliString, PauliString, size_t>;
    std::vector<TableauBasis> basis;
    std::vector<CircuitType> circuits;

  public:
    CliffordTable()=default;

    CliffordTable(std::function<bool(const QuantumCircuit&)> filter) : circuits() {
      std::vector<std::pair<PauliString, PauliString>> paulis;

      for (size_t s1 = 1; s1 < 16; s1++) {
        PauliString X, Z;
        X = PauliString::from_bitstring(2, s1);
        for (size_t s2 = 1; s2 < 16; s2++) {
          Z = PauliString::from_bitstring(2, s2);
          
          // Z should anticommute with X
          if (Z.commutes(X)) {
            continue;
          }

          paulis.push_back({ X,  Z });
          paulis.push_back({ X, -Z });
          paulis.push_back({-X,  Z });
          paulis.push_back({-X, -Z });
        }
      }

      Qubits qubits{0, 1};
      for (const auto& [X, Z] : paulis) {
        for (size_t r = 0; r < 24; r++) {
          QuantumCircuit qc(2);
          reduce_paulis(X, Z, qubits, qc);
          single_qubit_clifford_impl(qc, 0, r);
          if (filter(qc)) {
            if constexpr (std::is_same_v<CircuitType, QuantumCircuit>) {
              circuits.push_back(qc);
            } else if constexpr (std::is_same_v<CircuitType, Eigen::MatrixXcd>) {
              circuits.push_back(qc.to_matrix());
            } else {
              static_assert(std::is_same_v<CircuitType, QuantumCircuit> || std::is_same_v<CircuitType, Eigen::MatrixXcd>,
                  "Unsupported tempalte parameter passed to CliffordTable.");
            }

            basis.push_back(std::make_tuple(X, Z, r));
          }
        }
      }
    }

    CliffordTable(const CliffordTable& other) : basis(other.basis), circuits(other.circuits) {}

    size_t num_elements() const {
      return circuits.size();
    }

    CircuitType get_circuit(uint32_t r) const {
      return circuits[r];
    }

    std::vector<CircuitType> get_circuits() const {
      return circuits;
    }

    template <typename... Args>
    void apply_random(const Qubits& qubits, Args&... args) {
      uint32_t r = randi(0, circuits.size());
      if constexpr (std::is_same_v<CircuitType, QuantumCircuit>) {
        QuantumCircuit qc = circuits[r];
        qc.apply(qubits, args...);
      } else if constexpr (std::is_same_v<CircuitType, Eigen::MatrixXcd>) {
        Eigen::MatrixXcd matrix = circuits[r];

        ([&] {
         if constexpr (std::is_same_v<std::decay_t<Args>, QuantumCircuit>) {
           args.add_gate(matrix, qubits);
         } else {
           args.evolve(matrix, qubits);
         }
        }(), ...);
      }
    } 
};

