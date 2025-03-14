#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <unordered_set>
#include <vector>
#include <variant>
#include <complex>
#include <memory>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "PauliString.hpp"
#include "CircuitUtils.h"
#include "QuantumState/utils.hpp"

// --- Definitions for gates/measurements --- //

namespace gates {
  constexpr double sqrt2i_ = 0.707106781186547524400844362104849;
  constexpr std::complex<double> i_ = std::complex<double>(0.0, 1.0);

  struct H { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << sqrt2i_, sqrt2i_, sqrt2i_, -sqrt2i_).finished(); };

  struct I { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, 1.0).finished(); };
  struct X { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished(); };
  struct Y { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, -i_, i_, 0.0).finished(); };
  struct Z { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, -1.0).finished(); };

  struct sqrtX { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (1.0 - i_)/2.0, (1.0 - i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtY { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (-1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtZ { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, i_).finished(); };

  struct sqrtXd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtYd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 - i_)/2.0, (-1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtZd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, -i_).finished(); };

  struct T { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, sqrt2i_*(1.0 + i_)).finished(); };
  struct Td { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, sqrt2i_*(1.0 - i_)).finished(); };

  struct CX { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, 1, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, 1, 0, 0).finished(); };
  struct CY { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, -i_, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, i_, 0, 0).finished(); };
  struct CZ { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1).finished(); };
  struct SWAP { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished(); };
}

class Gate {
  public:
    Qubits qubits;
    uint32_t num_qubits;

    Gate(const Qubits& qubits)
      : qubits(qubits), num_qubits(qubits.size()) {
        if (!qargs_unique(qubits)) {
          throw std::runtime_error(fmt::format("Qubits {} provided to gate not unique.", qubits));
        }
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
          return gates::H::value;
        case SymbolicGate::GateLabel::X:
          return gates::X::value;
        case SymbolicGate::GateLabel::Y:
          return gates::Y::value;
        case SymbolicGate::GateLabel::Z:
          return gates::Z::value;
        case SymbolicGate::GateLabel::sqrtX:
          return gates::sqrtX::value;
        case SymbolicGate::GateLabel::sqrtY:
          return gates::sqrtY::value;
        case SymbolicGate::GateLabel::S:
          return gates::sqrtZ::value;
        case SymbolicGate::GateLabel::sqrtXd:
          return gates::sqrtXd::value;
        case SymbolicGate::GateLabel::sqrtYd:
          return gates::sqrtYd::value;
        case SymbolicGate::GateLabel::Sd:
          return gates::sqrtZd::value;
        case SymbolicGate::GateLabel::T:
          return gates::T::value;
        case SymbolicGate::GateLabel::Td:
          return gates::Td::value;
        case SymbolicGate::GateLabel::CX:
          return gates::CX::value;
        case SymbolicGate::GateLabel::CY:
          return gates::CY::value;
        case SymbolicGate::GateLabel::CZ:
          return gates::CZ::value;
        case SymbolicGate::GateLabel::SWAP:
          return gates::SWAP::value;
        default:
          return gates::I::value;
      }
    }

  public:
    SymbolicGate::GateLabel type;

    SymbolicGate(SymbolicGate::GateLabel type, const Qubits& qubits) : Gate(qubits), type(type) {
      if (num_qubits_for_gate(type) != qubits.size()) {
        throw std::runtime_error("Invalid qubits provided to SymbolicGate.");
      }
    }

    SymbolicGate(const char* name, const Qubits& qubits) : SymbolicGate(parse_gate(name), qubits) { }
    SymbolicGate(const std::string& name, const Qubits& qubits) : SymbolicGate(name.c_str(), qubits) { }

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
      
      return std::shared_ptr<Gate>(new SymbolicGate(new_type, qubits));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new SymbolicGate(type, qubits)); 
    }
};

class MatrixGate : public Gate {
  public:
    Eigen::MatrixXcd data;
    std::string label_str;

    MatrixGate(const Eigen::MatrixXcd& data, const Qubits& qubits, const std::string& label_str)
      : Gate(qubits), data(data), label_str(label_str) {}

    MatrixGate(const Eigen::MatrixXcd& data, const Qubits& qubits)
      : MatrixGate(data, qubits, "U") {}


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
      return std::shared_ptr<Gate>(new MatrixGate(data.adjoint(), qubits));
    }

    virtual bool is_clifford() const override {
      // No way to easily check if arbitrary data is Clifford at the moment
      return false;
    }

    virtual std::shared_ptr<Gate> clone() override { 
      return std::shared_ptr<Gate>(new MatrixGate(data, qubits)); 
    }
};

#define GATECLONE(A) 								                    \
virtual std::shared_ptr<Gate> clone() override { 	      \
  return std::shared_ptr<Gate>(new A(this->qubits)); 	  \
}

#define GATE_PI 3.14159265359

class RxRotationGate : public Gate {
  private:
    bool adj;

  public:
    RxRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Rx gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override { 
      return 1;
    }

    virtual std::string label() const override { 
      return adj ? "Rxd" : "Rx";
    }

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != num_params()) {
        std::string error_message = "Invalid number of params passed to define(). Expected " 
                                   + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
        throw std::invalid_argument(error_message);
      }

      Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

      double t = params[0];
      gate << std::complex<double>(std::cos(t/2), 0), std::complex<double>(0, -std::sin(t/2)), 
              std::complex<double>(0, -std::sin(t/2)), std::complex<double>(std::cos(t/2), 0);

      if (adj) {
        gate = gate.adjoint();
      }

      return gate;
    }

    virtual bool is_clifford() const override {
      return false;
    }

    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new RxRotationGate(qubits, !adj));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new RxRotationGate(qubits, adj));
    }
};

class RyRotationGate : public Gate {
  private:
    bool adj;

  public:
    RyRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Ry gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override { 
      return 1; 
    }
    
    virtual std::string label() const override { 
      return adj ? "Ryd" : "Ry";
    }

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != num_params()) {
        std::string error_message = "Invalid number of params passed to define(). Expected " 
                                   + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
        throw std::invalid_argument(error_message);
      }

      Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

      double t = params[0];
      gate << std::complex<double>(std::cos(t/2), 0), std::complex<double>(-std::sin(t/2), 0), 
              std::complex<double>(std::sin(t/2), 0), std::complex<double>(std::cos(t/2), 0);

      if (adj) {
        gate = gate.adjoint();
      }

      return gate;
    }

    virtual bool is_clifford() const override {
      return false;
    }

    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new RyRotationGate(qubits, !adj));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new RyRotationGate(qubits, adj));
    }
};

class RzRotationGate : public Gate {
  private:
    bool adj;

  public:
    RzRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Rz gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override {
      return 1; 
    }

    virtual std::string label() const override { 
      return adj ? "Rzd" : "Rz"; 
    }

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != num_params()) {
        std::string error_message = "Invalid number of params passed to define(). Expected " 
                                   + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
        throw std::invalid_argument(error_message);
      }

      Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

      double t = params[0];
      gate << std::complex<double>(std::cos(t/2), -std::sin(t/2)), std::complex<double>(0.0, 0.0), 
              std::complex<double>(0.0, 0.0), std::complex<double>(std::cos(t/2), std::sin(t/2));

      if (adj) {
        gate = gate.adjoint();
      }

      return gate;
    }

    virtual bool is_clifford() const override {
      return false;
    }

    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new RzRotationGate(qubits, !adj));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new RzRotationGate(qubits, adj));
    }
};

template <class GateType>
class MemoizedGate : public GateType {
  private:
    bool adj;

    static std::vector<Eigen::MatrixXcd> memoized_gates;
    static bool defined;

    static void generate_memoized_gates(uint32_t res, double min, double max) {
      MemoizedGate<GateType>::memoized_gates = std::vector<Eigen::MatrixXcd>(res);
      double bin_width = (max - min)/res;
      Qubits qubits{0};

      for (uint32_t i = 0; i < res; i++) {
        double d = min + bin_width*i;

        GateType gate(qubits);
        std::vector<double> params{d};
        MemoizedGate<GateType>::memoized_gates[i] = gate.define(params);
      }

      MemoizedGate<GateType>::defined = true;
    }

    static uint32_t get_idx(double d, uint32_t res, double min, double max) {
      double dt = std::fmod(d, 2*GATE_PI);

      double bin_width = static_cast<double>(max - min)/res;
      return static_cast<uint32_t>((dt - min)/bin_width);
    }

  public:
    MemoizedGate(const Qubits& qubits, bool adj=false) : GateType(qubits), adj(adj) {}

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (!MemoizedGate<GateType>::defined) {
        MemoizedGate<GateType>::generate_memoized_gates(200, 0, 2*GATE_PI);
      }

      if (params.size() != this->num_params()) {
        std::string error_message = "Invalid number of params passed to define(). Expected " 
                                   + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
        throw std::invalid_argument(error_message);
      }

      double d = params[0];
      uint32_t idx = MemoizedGate<GateType>::get_idx(d, 200, 0, 2*GATE_PI);

      Eigen::MatrixXcd g = MemoizedGate<GateType>::memoized_gates[idx];

      if (adj) {
        g = g.adjoint();
      }

      return g;
    }

    virtual bool is_clifford() const override {
      return false;
    }
    
    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new MemoizedGate<GateType>(this->qubits, !adj));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new MemoizedGate<GateType>(this->qubits, adj));
    }
};

template <class GateType>
std::vector<Eigen::MatrixXcd> MemoizedGate<GateType>::memoized_gates;

template <class GateType>
bool MemoizedGate<GateType>::defined = false;

static std::shared_ptr<Gate> parse_gate(const std::string& s, const Qubits& qubits) {
  if (s == "H" || s == "h") {
    return std::make_shared<MatrixGate>(gates::H::value, qubits, "h");
  } else if (s == "X" || s == "x") {
    return std::make_shared<MatrixGate>(gates::X::value, qubits, "x");
  } else if (s == "Y" || s == "y") {
    return std::make_shared<MatrixGate>(gates::Y::value, qubits, "y");
  } else if (s == "Z" || s == "z") {
    return std::make_shared<MatrixGate>(gates::Z::value, qubits, "z");
  } else if (s == "RX" || s == "Rx" || s == "rx") {
    return std::make_shared<RxRotationGate>(qubits);
  } else if (s == "RXM" || s == "Rxm" || s == "rxm") {
    return std::make_shared<MemoizedGate<RxRotationGate>>(qubits);
  } else if (s == "RY" || s == "Ry" || s == "ry") {
    return std::make_shared<RyRotationGate>(qubits);
  } else if (s == "RYM" || s == "Rym" || s == "rym") {
    return std::make_shared<MemoizedGate<RyRotationGate>>(qubits);
  } else if (s == "RZ" || s == "Rz" || s == "rz") {
    return std::make_shared<RzRotationGate>(qubits);
  } else if (s == "RZM" || s == "Rzm" || s == "rzm") {
    return std::make_shared<MemoizedGate<RzRotationGate>>(qubits);
  } else if (s == "CX" || s == "cx") {
    return std::make_shared<MatrixGate>(gates::CX::value, qubits, "cx");
  } else if (s == "CY" || s == "cy") {
    return std::make_shared<MatrixGate>(gates::CY::value, qubits, "cy");
  } else if (s == "CZ" || s == "cz") {
    return std::make_shared<MatrixGate>(gates::CZ::value, qubits, "cz");
  } else if (s == "swap" || s == "SWAP") {
    return std::make_shared<MatrixGate>(gates::SWAP::value, qubits, "swap");
  } else {
    throw std::invalid_argument(fmt::format("Invalid gate type: {}", s));
  }
}

class PauliString;

struct Measurement {
  Qubits qubits;
  std::optional<PauliString> pauli;
  std::optional<bool> outcome;

  Measurement(const Qubits& qubits, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
  static Measurement computational_basis(uint32_t q, std::optional<bool> outcome=std::nullopt);
  PauliString get_pauli() const;
  bool is_basis() const;
  bool is_forced() const;
  bool get_outcome() const;
};

struct WeakMeasurement {
  Qubits qubits;
  double beta;
  std::optional<PauliString> pauli;
  std::optional<bool> outcome;

  WeakMeasurement(const Qubits& qubits, double beta, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
  PauliString get_pauli() const;
  bool is_forced() const;
  bool get_outcome() const;
};

typedef std::variant<std::shared_ptr<Gate>, Measurement, WeakMeasurement> Instruction;

static Instruction copy_instruction(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](std::shared_ptr<Gate> gate) {
      return Instruction(gate->clone());
    },
    [](Measurement m) {
      return Instruction(Measurement(m.qubits, m.pauli, m.outcome));
    },
    [](WeakMeasurement m) {
      return Instruction(WeakMeasurement(m.qubits, m.beta, m.pauli, m.outcome));
    }
  }, inst);
}
