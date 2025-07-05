#pragma once

#include <random>
#include "CircuitUtils.h"
#include "Instructions.hpp"

#include <iostream>

#include <fmt/format.h>

// --- Definitions for QuantumCircuit --- //

class QuantumCircuit {
  private:
    uint32_t num_qubits;

  public:
    std::vector<Instruction> instructions;

    QuantumCircuit() : num_qubits(0) {}

    QuantumCircuit(uint32_t num_qubits) : num_qubits(num_qubits) {}

    QuantumCircuit(const QuantumCircuit& qc) : num_qubits(qc.num_qubits) { 
      append(qc); 
    };

    uint32_t get_num_qubits() const {
      return num_qubits;
    }

    void resize(uint32_t num_qubits) {
      this->num_qubits = num_qubits;
    }

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& stream, const QuantumCircuit& qc) {
      stream << qc.to_string();
      return stream;
    }

    uint32_t num_params() const;
    uint32_t length() const;

    bool is_unitary() const;
    bool is_clifford() const;

    template <typename... QuantumStates>
    void apply(QuantumStates&... states) const {
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.append(*this);
       } else {
         states.evolve(*this);
       }
      }(), ...);
    }

    template <typename... QuantumStates>
    void apply(const Qubits& qubits, QuantumStates&... states) const {
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.append(*this, qubits);
       } else {
         states.evolve(*this, qubits);
       }
      }(), ...);
    }

    void apply_qubit_map(const Qubits& qubits);

    Qubits get_support() const;
    std::pair<QuantumCircuit, Qubits> reduce() const;

    void validate_instruction(const Instruction& inst) const;

    void add_instruction(const Instruction& inst);
    void add_measurement(const Measurement& m);
    void add_measurement(const Qubits& qubits, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt) {
      Measurement m(qubits, pauli, outcome);
      add_measurement(m);
    }
    void add_weak_measurement(const WeakMeasurement& m);
    void add_weak_measurement(const Qubits& qubits, double beta, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt) {
      WeakMeasurement m(qubits, beta, pauli, outcome);
      add_weak_measurement(m);
    }

    void add_gate(const std::string& name, const Qubits& qubits);
    void add_gate(const std::shared_ptr<Gate> &gate);
    void add_gate(const Eigen::MatrixXcd& gate, const Qubits& qubits);
    void add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit);

    void h(uint32_t q) {
      add_gate("h", {q});
    }

    void s(uint32_t q) {
      add_gate("s", {q});
    }

    void sd(uint32_t q) {
      add_gate("sd", {q});
    }

    void t(uint32_t q) {
      add_gate("t", {q});
    }

    void td(uint32_t q) {
      add_gate("td", {q});
    }

    void x(uint32_t q) {
      add_gate("x", {q});
    }

    void y(uint32_t q) {
      add_gate("y", {q});
    }

    void z(uint32_t q) {
      add_gate("z", {q});
    }

    void sqrtX(uint32_t q) {
      add_gate("sqrtX", {q});
    }

    void sqrtY(uint32_t q) {
      add_gate("sqrtY", {q});
    }

    void sqrtZ(uint32_t q) {
      add_gate("sqrtZ", {q});
    }

    void sqrtXd(uint32_t q) {
      add_gate("sqrtXd", {q});
    }

    void sqrtYd(uint32_t q) {
      add_gate("sqrtYd", {q});
    }

    void sqrtZd(uint32_t q) {
      add_gate("sqrtZd", {q});
    }

    void cx(uint32_t q1, uint32_t q2) {
      add_gate("cx", {q1, q2});
    }

    void cy(uint32_t q1, uint32_t q2) {
      add_gate("cy", {q1, q2});
    }

    void cz(uint32_t q1, uint32_t q2) {
      add_gate("cz", {q1, q2});
    }

    void swap(uint32_t q1, uint32_t q2) {
      add_gate("swap", {q1, q2});
    }

    void random_clifford(const Qubits& qubits);

    void append(const QuantumCircuit& other);
    void append(const QuantumCircuit& other, const Qubits& qubits);
    void append(const Instruction& inst);

    QuantumCircuit bind_params(const std::vector<double>& params) const;
    size_t get_num_measurements() const;
    void set_measurement_outcomes(const std::vector<bool>& outcomes);

    QuantumCircuit adjoint(const std::optional<std::vector<double>>& params_opt = std::nullopt) const;
    QuantumCircuit reverse() const;
    QuantumCircuit conjugate(const QuantumCircuit& other) const;

    std::vector<QuantumCircuit> split_into_unitary_components() const;

    Eigen::MatrixXcd to_matrix(const std::optional<std::vector<double>>& params_opt = std::nullopt) const;
};

template <>
struct fmt::formatter<QuantumCircuit> {
  template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
    auto format(const QuantumCircuit& qc, FormatContext& ctx) {
      return format_to(ctx.out(), "{}", qc.to_string());
    }
};


// --- Library for building common circuits --- //

QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc=true);
QuantumCircuit hardware_efficient_ansatz(uint32_t num_qubits, uint32_t depth, const std::vector<std::string>& rotation_gates, const std::string& entangling_gate = "cz", bool final_layer = true);
QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<Qubits>& qargs_opt = std::nullopt);
QuantumCircuit random_clifford(uint32_t num_qubits);
