#pragma once

#include "Instructions.hpp"

// Defining types of gates

#define GATECLONE(A) 								              \
virtual std::shared_ptr<Gate> clone() override { 	\
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
  Eigen::MatrixXcd data(1u << qubits.size(), 1u << qubits.size());
  std::complex<double> i(0.0, 1.0);
  double sqrt2 = 1.41421356237;

  if (s == "H" || s == "h") {
    data << 1.0, 1.0, 1.0, -1.0;
    data /= sqrt2;
    return std::make_shared<MatrixGate>(data, qubits, "h");
  } else if (s == "X" || s == "x") {
    data << 0.0, 1.0, 1.0, 0.0;
    return std::make_shared<MatrixGate>(data, qubits, "x");
  } else if (s == "Y" || s == "y") {
    data << 0.0, 1.0*i, -1.0*i, 0.0;
    return std::make_shared<MatrixGate>(data, qubits);
  } else if (s == "Z" || s == "z") {
    data << 1.0, 0.0, 0.0, -1.0;
    return std::make_shared<MatrixGate>(data, qubits);
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
  } else if (s == "CZ" || s == "cz") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, -1.0;
    return std::make_shared<MatrixGate>(data, qubits, "cz");
  } else if (s == "CX" || s == "cx") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0;
    return std::make_shared<MatrixGate>(data, qubits, "cx");
  } else if (s == "CY" || s == "cy") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0*i,
            0.0, 0.0, 1.0*i, 0.0;
    return std::make_shared<MatrixGate>(data, qubits, "cy");
  } else if (s == "swap" || s == "SWAP") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
    return std::make_shared<MatrixGate>(data, qubits, "swap");
  } else {
    throw std::invalid_argument("Invalid gate type: " + s);
  }
}
