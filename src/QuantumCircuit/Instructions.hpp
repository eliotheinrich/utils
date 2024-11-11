#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <unordered_set>
#include <vector>
#include <variant>
#include <string>
#include <complex>
#include <memory>
#include <assert.h>

#include "CircuitUtils.h"

#include <fmt/format.h>

#include "QuantumState/utils.hpp"

// --- Definitions for gates/measurements --- //

class Gate {
  public:
    std::vector<uint32_t> qbits;
    uint32_t num_qubits;

    Gate(const std::vector<uint32_t>& qbits)
      : qbits(qbits), num_qubits(qbits.size()) {
        //assert(qargs_unique(qbits));
      }

    virtual uint32_t num_params() const=0;

    virtual std::string label() const=0;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const=0;

    Eigen::MatrixXcd define() const {
      if (num_params() > 0) {
        throw std::invalid_argument("Unbound parameters; cannot define gate.");
      }

      return define(std::vector<double>());
    }

    virtual std::shared_ptr<Gate> adjoint() const=0;

    Eigen::MatrixXcd adjoint(const std::vector<double>& params) const {
      return adjoint()->define(params);
    }

    virtual bool is_clifford() const=0;

    virtual std::shared_ptr<Gate> clone()=0;
};

class SymbolicGate : public Gate {
  private:
    enum GateLabel {
      H, X, Y, Z, sqrtX, sqrtY, S, sqrtXd, sqrtYd, Sd, T, Td, CX, CY, CZ, SWAP
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> clifford_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::H, 
      SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::S, 
      SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::Sd, 
      SymbolicGate::GateLabel::X, SymbolicGate::GateLabel::Y, SymbolicGate::GateLabel::Z, 
      SymbolicGate::GateLabel::CX, SymbolicGate::GateLabel::CY, SymbolicGate::GateLabel::CZ, 
      SymbolicGate::GateLabel::SWAP
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> single_qubit_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::H, SymbolicGate::GateLabel::T, SymbolicGate::GateLabel::Td,
      SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::S, 
      SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::Sd, 
      SymbolicGate::GateLabel::X, SymbolicGate::GateLabel::Y, SymbolicGate::GateLabel::Z
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> two_qubit_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::CX, SymbolicGate::GateLabel::CY, SymbolicGate::GateLabel::CZ, 
      SymbolicGate::GateLabel::SWAP
    };

    inline static std::unordered_map<SymbolicGate::GateLabel, SymbolicGate::GateLabel> adjoint_map = std::unordered_map<SymbolicGate::GateLabel, SymbolicGate::GateLabel>{
      {SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtXd},
      {SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::sqrtYd},
      {SymbolicGate::GateLabel::S, SymbolicGate::GateLabel::Sd},
      {SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtX},
      {SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::sqrtY},
      {SymbolicGate::GateLabel::Sd, SymbolicGate::GateLabel::S},
      {SymbolicGate::GateLabel::T, SymbolicGate::GateLabel::Td},
      {SymbolicGate::GateLabel::Td, SymbolicGate::GateLabel::T},
    };

    static constexpr bool str_equal_ci(const char* a, const char* b) {
      while (*a && *b) {
        if (std::tolower(*a) != std::tolower(*b)) {
          return false;
        }
        ++a;
        ++b;
      }
      return *a == '\0' && *b == '\0';
    }

    constexpr SymbolicGate::GateLabel parse_gate(const char* name) const {
      if (str_equal_ci(name, "h")) {
        return SymbolicGate::GateLabel::H;
      } else if (str_equal_ci(name, "x")) {
        return SymbolicGate::GateLabel::X;
      } else if (str_equal_ci(name, "y")) {
        return SymbolicGate::GateLabel::Y;
      } else if (str_equal_ci(name, "z")) {
        return SymbolicGate::GateLabel::Z;
      } else if (str_equal_ci(name, "sqrtx")) {
        return SymbolicGate::GateLabel::sqrtX;
      } else if (str_equal_ci(name, "sqrty")) {
        return SymbolicGate::GateLabel::sqrtY;
      } else if (str_equal_ci(name, "sqrtz") || str_equal_ci(name, "s")) {
        return SymbolicGate::GateLabel::S;
      } else if (str_equal_ci(name, "sqrtxd")) {
        return SymbolicGate::GateLabel::sqrtXd;
      } else if (str_equal_ci(name, "sqrtyd")) {
        return SymbolicGate::GateLabel::sqrtYd;
      } else if (str_equal_ci(name, "sqrtzd") || str_equal_ci(name, "sd")) {
        return SymbolicGate::GateLabel::Sd;
      } else if (str_equal_ci(name, "t")) {
        return SymbolicGate::GateLabel::T;
      } else if (str_equal_ci(name, "td")) {
        return SymbolicGate::GateLabel::Td;
      } else if (str_equal_ci(name, "cx")) {
        return SymbolicGate::GateLabel::CX;
      } else if (str_equal_ci(name, "cy")) {
        return SymbolicGate::GateLabel::CY;
      } else if (str_equal_ci(name, "cz")) {
        return SymbolicGate::GateLabel::CZ;
      } else if (str_equal_ci(name, "swap")) {
        return SymbolicGate::GateLabel::SWAP;
      } else {
        throw std::runtime_error(fmt::format("Error: unknown gate {}.", name));
      }
    }

    constexpr const char* type_to_string(SymbolicGate::GateLabel g) const {
      switch (g) {
        case SymbolicGate::GateLabel::H:
          return "H";
        case SymbolicGate::GateLabel::X:
          return "X";
        case SymbolicGate::GateLabel::Y:
          return "Y";
        case SymbolicGate::GateLabel::Z:
          return "Z";
        case SymbolicGate::GateLabel::sqrtX:
          return "sqrtX";
        case SymbolicGate::GateLabel::sqrtY:
          return "sqrtY";
        case SymbolicGate::GateLabel::S:
          return "S";
        case SymbolicGate::GateLabel::sqrtXd:
          return "sqrtXd";
        case SymbolicGate::GateLabel::sqrtYd:
          return "sqrtYd";
        case SymbolicGate::GateLabel::Sd:
          return "Sd";
        case SymbolicGate::GateLabel::T:
          return "T";
        case SymbolicGate::GateLabel::Td:
          return "Td";
        case SymbolicGate::GateLabel::CX:
          return "CX";
        case SymbolicGate::GateLabel::CY:
          return "CY";
        case SymbolicGate::GateLabel::CZ:
          return "CZ";
        case SymbolicGate::GateLabel::SWAP:
          return "SWAP";
        default:
          throw std::runtime_error("Invalid gate type.");
      }
    }

    constexpr size_t num_qubits_for_gate(SymbolicGate::GateLabel g) const {
      switch (g) {
        case SymbolicGate::GateLabel::H:
          return 1;
        case SymbolicGate::GateLabel::X:
          return 1;
        case SymbolicGate::GateLabel::Y:
          return 1;
        case SymbolicGate::GateLabel::Z:
          return 1;
        case SymbolicGate::GateLabel::sqrtX:
          return 1;
        case SymbolicGate::GateLabel::sqrtY:
          return 1;
        case SymbolicGate::GateLabel::S:
          return 1;
        case SymbolicGate::GateLabel::sqrtXd:
          return 1;
        case SymbolicGate::GateLabel::sqrtYd:
          return 1;
        case SymbolicGate::GateLabel::Sd:
          return 1;
        case SymbolicGate::GateLabel::T:
          return 1;
        case SymbolicGate::GateLabel::Td:
          return 1;
        case SymbolicGate::GateLabel::CX:
          return 2;
        case SymbolicGate::GateLabel::CY:
          return 2;
        case SymbolicGate::GateLabel::CZ:
          return 2;
        case SymbolicGate::GateLabel::SWAP:
          return 2;
        default:
          throw std::runtime_error("Invalid gate type.");
        
      }
    }

    Eigen::MatrixXcd to_data(SymbolicGate::GateLabel g) const {
      switch (g) {
        case SymbolicGate::GateLabel::H:
          return quantumstate_utils::H::value;
        case SymbolicGate::GateLabel::X:
          return quantumstate_utils::X::value;
        case SymbolicGate::GateLabel::Y:
          return quantumstate_utils::Y::value;
        case SymbolicGate::GateLabel::Z:
          return quantumstate_utils::Z::value;
        case SymbolicGate::GateLabel::sqrtX:
          return quantumstate_utils::sqrtX::value;
        case SymbolicGate::GateLabel::sqrtY:
          return quantumstate_utils::sqrtY::value;
        case SymbolicGate::GateLabel::S:
          return quantumstate_utils::sqrtZ::value;
        case SymbolicGate::GateLabel::sqrtXd:
          return quantumstate_utils::sqrtXd::value;
        case SymbolicGate::GateLabel::sqrtYd:
          return quantumstate_utils::sqrtYd::value;
        case SymbolicGate::GateLabel::Sd:
          return quantumstate_utils::sqrtZd::value;
        case SymbolicGate::GateLabel::T:
          return quantumstate_utils::T::value;
        case SymbolicGate::GateLabel::Td:
          return quantumstate_utils::Td::value;
        case SymbolicGate::GateLabel::CX:
          return quantumstate_utils::CX::value;
        case SymbolicGate::GateLabel::CY:
          return quantumstate_utils::CY::value;
        case SymbolicGate::GateLabel::CZ:
          return quantumstate_utils::CZ::value;
        case SymbolicGate::GateLabel::SWAP:
          return quantumstate_utils::SWAP::value;
        default:
          return quantumstate_utils::I::value;
      }
    }

  public:
    SymbolicGate::GateLabel type;

    SymbolicGate(SymbolicGate::GateLabel type, const std::vector<uint32_t>& qbits) : Gate(qbits), type(type) {
      if (num_qubits_for_gate(type) != qbits.size()) {
        throw std::runtime_error("Invalid qubits provided to SymbolicGate.");
      }
    }

    SymbolicGate(const char* name, const std::vector<uint32_t>& qbits) : SymbolicGate(parse_gate(name), qbits) { }
    SymbolicGate(const std::string& name, const std::vector<uint32_t>& qbits) : SymbolicGate(name.c_str(), qbits) { }

    virtual bool is_clifford() const override {
      return SymbolicGate::clifford_gates.contains(type);
    }

    virtual uint32_t num_params() const override {
      return 0;
    }

    virtual std::string label() const override {
      return type_to_string(type);
    }

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != 0) {
        throw std::invalid_argument("Cannot pass parameters to SymbolicGate.");
      }

      return to_data(type);
    }

    virtual std::shared_ptr<Gate> adjoint() const override {
      SymbolicGate::GateLabel new_type;
      if (SymbolicGate::adjoint_map.contains(type)) {
        new_type = SymbolicGate::adjoint_map[type];
      } else {
        new_type = type;
      }
      
      return std::shared_ptr<Gate>(new SymbolicGate(new_type, qbits));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new SymbolicGate(type, qbits)); 
    }
};

class MatrixGate : public Gate {
  public:
    Eigen::MatrixXcd data;
    std::string label_str;

    MatrixGate(const Eigen::MatrixXcd& data, const std::vector<uint32_t>& qbits, const std::string& label_str)
      : Gate(qbits), data(data), label_str(label_str) {}

    MatrixGate(const Eigen::MatrixXcd& data, const std::vector<uint32_t>& qbits)
      : MatrixGate(data, qbits, "U") {}


    virtual uint32_t num_params() const override {
      return 0;
    }

    virtual std::string label() const override {
      return label_str;
    }

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != 0) {
        throw std::invalid_argument("Cannot pass parameters to MatrixGate.");
      }

      return data;
    }

    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new MatrixGate(data.adjoint(), qbits));
    }

    virtual bool is_clifford() const override {
      // No way to easily check if arbitrary data is Clifford at the moment
      return false;
    }

    virtual std::shared_ptr<Gate> clone() override { 
      return std::shared_ptr<Gate>(new MatrixGate(data, qbits)); 
    }
};

struct Measurement {
  std::vector<uint32_t> qbits;
  Measurement(const std::vector<uint32_t>& qbits) : qbits(qbits) {}
};

typedef std::variant<std::shared_ptr<Gate>, Measurement> Instruction;
