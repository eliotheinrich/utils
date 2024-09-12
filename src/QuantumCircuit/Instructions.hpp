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
#include <fmt/format.h>

#include "CircuitUtils.h"

// --- Definitions for gates/measurements --- //

class Gate {
  public:
    std::vector<uint32_t> qbits;
    uint32_t num_qubits;

    Gate(const std::vector<uint32_t>& qbits)
      : qbits(qbits), num_qubits(qbits.size()) {
        assert(qargs_unique(qbits));
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

    SymbolicGate::GateLabel parse_gate(const std::string& name) const {
      std::string s = name;
      std::transform(s.begin(), s.end(), s.begin(),
          [](unsigned char c){ return std::tolower(c); });

      if (s == "h") {
        return SymbolicGate::GateLabel::H;
      } else if (s == "x") {
        return SymbolicGate::GateLabel::X;
      } else if (s == "y") {
        return SymbolicGate::GateLabel::Y;
      } else if (s == "z") {
        return SymbolicGate::GateLabel::Z;
      } else if (s == "sqrtx") {
        return SymbolicGate::GateLabel::sqrtX;
      } else if (s == "sqrty") {
        return SymbolicGate::GateLabel::sqrtY;
      } else if (s == "sqrtz" || s == "s") {
        return SymbolicGate::GateLabel::S;
      } else if (s == "sqrtxd") {
        return SymbolicGate::GateLabel::sqrtXd;
      } else if (s == "sqrtyd") {
        return SymbolicGate::GateLabel::sqrtYd;
      } else if (s == "sqrtzd" || s == "sd") {
        return SymbolicGate::GateLabel::Sd;
      } else if (s == "t") {
        return SymbolicGate::GateLabel::T;
      } else if (s == "td") {
        return SymbolicGate::GateLabel::Td;
      } else if (s == "cx") {
        return SymbolicGate::GateLabel::CX;
      } else if (s == "cy") {
        return SymbolicGate::GateLabel::CY;
      } else if (s == "cz") {
        return SymbolicGate::GateLabel::CZ;
      } else if (s == "swap") {
        return SymbolicGate::GateLabel::SWAP;
      } else {
        throw std::runtime_error(fmt::format("Error: unknown gate {}.", name));
      }
    }

    size_t num_qubits_for_gate(SymbolicGate::GateLabel g) const {
      if (SymbolicGate::single_qubit_gates.contains(g)) {
        return 1;
      } else if (SymbolicGate::two_qubit_gates.contains(g)) {
          return 2;
      } else {
        throw std::runtime_error(fmt::format("Cannot determine number of qubits for gate {}", g));
      }
    }

    Eigen::MatrixXcd to_data(SymbolicGate::GateLabel g) const {
      size_t nqb = 1u << num_qubits_for_gate(g);
      Eigen::MatrixXcd data(nqb, nqb);

      static double sqrt2 = 0.70710678118;
      switch (g) {
        case SymbolicGate::GateLabel::H:
          data << sqrt2, sqrt2, sqrt2, -sqrt2;
          break;
        case SymbolicGate::GateLabel::X:
          data << 0, 1, 1, 0;
          break;
        case SymbolicGate::GateLabel::Y:
          data << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0;
          break;
        case SymbolicGate::GateLabel::Z:
          data << 1, 0, 0, -1;
          break;
        case SymbolicGate::GateLabel::sqrtX:
          data << std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5), std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5);
          break;
        case SymbolicGate::GateLabel::sqrtY:
          data << std::complex<double>(0.5, 0.5), std::complex<double>(-0.5, -0.5), std::complex<double>(0.5, 0.5), std::complex<double>(0.5, 0.5);
          break;
        case SymbolicGate::GateLabel::S:
          data << 1, 0, 0, std::complex<double>(0, 1);
          break;
        case SymbolicGate::GateLabel::sqrtXd:
          data << std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5), std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5);
          break;
        case SymbolicGate::GateLabel::sqrtYd:
          data << std::complex<double>(0.5, -0.5), std::complex<double>(0.5, -0.5), std::complex<double>(-0.5, 0.5), std::complex<double>(0.5, -0.5);
          break;
        case SymbolicGate::GateLabel::Sd:
          data << 1, 0, 0, std::complex<double>(0, -1);
          break;
        case SymbolicGate::GateLabel::T:
          data << 1, 0, 0, std::complex<double>(sqrt2, sqrt2);
          break;
        case SymbolicGate::GateLabel::Td:
          data << 1, 0, 0, std::complex<double>(sqrt2, -sqrt2);
          break;
        case SymbolicGate::GateLabel::CX:
          data << 1, 0, 0, 0, 
               0, 1, 0, 0, 
               0, 0, 0, 1, 
               0, 0, 1, 0;
          break;
        case SymbolicGate::GateLabel::CY:
          data << 1, 0, 0, 0, 
               0, 1, 0, 0, 
               0, 0, 0, std::complex<double>(0, -1), 
               0, 0, std::complex<double>(0, 1), 0;
          break;
        case SymbolicGate::GateLabel::CZ:
          data << 1, 0, 0, 0, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, -1;
          break;
        case SymbolicGate::GateLabel::SWAP:
          data << 1, 0, 0, 0, 
               0, 0, 1, 0, 
               0, 1, 0, 0, 
               0, 0, 0, 1;
          break;
      }

      return data;
    }

  public:
    SymbolicGate::GateLabel type;

    SymbolicGate(SymbolicGate::GateLabel type, const std::vector<uint32_t>& qbits) : Gate(qbits), type(type) {
      if (num_qubits_for_gate(type) != qbits.size()) {
        throw std::runtime_error("Invalid qubits provided to SymbolicGate.");
      }
    }

    SymbolicGate(const std::string& name, const std::vector<uint32_t>& qbits) : SymbolicGate(parse_gate(name), qbits) { }

    virtual bool is_clifford() const override {
      return SymbolicGate::clifford_gates.contains(type);
    }

    virtual uint32_t num_params() const override {
      return 0;
    }

    virtual std::string label() const override {
      switch (type) {
        case SymbolicGate::GateLabel::H:
          return "H";
          break;
        case SymbolicGate::GateLabel::X:
          return "X";
          break;
        case SymbolicGate::GateLabel::Y:
          return "Y";
          break;
        case SymbolicGate::GateLabel::Z:
          return "Z";
          break;
        case SymbolicGate::GateLabel::sqrtX:
          return "sqrtX";
          break;
        case SymbolicGate::GateLabel::sqrtY:
          return "sqrtY";
          break;
        case SymbolicGate::GateLabel::S:
          return "S";
          break;
        case SymbolicGate::GateLabel::sqrtXd:
          return "sqrtXd";
          break;
        case SymbolicGate::GateLabel::sqrtYd:
          return "sqrtYd";
          break;
        case SymbolicGate::GateLabel::Sd:
          return "Sd";
          break;
        case SymbolicGate::GateLabel::T:
          return "T";
          break;
        case SymbolicGate::GateLabel::Td:
          return "Td";
          break;
        case SymbolicGate::GateLabel::CX:
          return "CX";
          break;
        case SymbolicGate::GateLabel::CY:
          return "CY";
          break;
        case SymbolicGate::GateLabel::CZ:
          return "CZ";
          break;
        case SymbolicGate::GateLabel::SWAP:
          return "SWAP";
          break;
        default:
          throw std::runtime_error("Invalid gate type.");
          break;
      }
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
