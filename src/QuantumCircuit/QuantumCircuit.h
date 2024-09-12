#pragma once

#include <random>
#include "CircuitUtils.h"
#include "Gates.hpp"


#include <fmt/format.h>

#include <iostream>

// --- Definitions for QuantumCircuit --- //

class QuantumCircuit {
  public:
    uint32_t num_qubits;
    std::vector<Instruction> instructions;

    QuantumCircuit() : num_qubits(0) {}

    QuantumCircuit(uint32_t num_qubits) : num_qubits(num_qubits) {}

    QuantumCircuit(const QuantumCircuit& qc) : num_qubits(qc.num_qubits) { 
      append(qc); 
    };

    void resize(uint32_t num_qubits) {
      this->num_qubits = num_qubits;
    }

    std::string to_string() const;

    uint32_t num_params() const;
    uint32_t length() const;

    bool contains_measurement() const;

    void apply_qubit_map(const std::vector<uint32_t>& qubits);

    void add_instruction(const Instruction& inst);
    void add_measurement(uint32_t qubit);
    void add_measurement(const std::vector<uint32_t>& qargs);
    void add_measurement(const Measurement& m);
    void add_gate(const std::string& name, const std::vector<uint32_t>& qbits);
    void add_gate(const std::shared_ptr<Gate> &gate);
    void add_gate(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits);
    void add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit);

    void append(const QuantumCircuit& other);
    void append(const QuantumCircuit& other, const std::vector<uint32_t>& qbits);
    void append(const Instruction& inst);

    bool is_clifford() const;

    QuantumCircuit bind_params(const std::vector<double>& params) const;

    QuantumCircuit adjoint(const std::optional<std::vector<double>>& params_opt = std::nullopt) const;

    Eigen::MatrixXcd to_matrix(const std::optional<std::vector<double>>& params_opt = std::nullopt) const;
};

// --- Library for building common circuits --- //

QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc=true, std::optional<int> seed = std::nullopt);
QuantumCircuit hardware_efficient_ansatz(uint32_t num_qubits, uint32_t depth, const std::vector<std::string>& rotation_gates, const std::string& entangling_gate = "cz", bool final_layer = true);
QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<std::vector<uint32_t>>& qargs_opt = std::nullopt);
QuantumCircuit random_clifford(uint32_t num_qubits, std::minstd_rand& rng);
