#pragma once

#include "Instructions.hpp"

// Defining types of gates

#define GATECLONE(A) 								              \
virtual std::shared_ptr<Gate> clone() override { 	\
  return std::shared_ptr<Gate>(new A(this->qbits)); 	  \
}

#define GATE_PI 3.14159265359

class RxRotationGate : public Gate {
  public:
    RxRotationGate(const std::vector<uint32_t>& qbits) : Gate(qbits) {
      if (qbits.size() != 1) {
        std::string error_message = "Rx gate can only have a single qubit. Passed " + std::to_string(qbits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override { 
      return 1;
    }

    virtual std::string label() const override { 
      return "Rx";
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
      return gate;
    }

    GATECLONE(RxRotationGate);
};

class RyRotationGate : public Gate {
  public:
    RyRotationGate(const std::vector<uint32_t>& qbits) : Gate(qbits) {
      if (qbits.size() != 1) {
        std::string error_message = "Ry gate can only have a single qubit. Passed " + std::to_string(qbits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override { 
      return 1; 
    }
    
    virtual std::string label() const override { 
      return "Ry"; 
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
      return gate;
    }

    GATECLONE(RyRotationGate);
};

class RzRotationGate : public Gate {
  public:
    RzRotationGate(const std::vector<uint32_t>& qbits) : Gate(qbits) {
      if (qbits.size() != 1) {
        std::string error_message = "Rz gate can only have a single qubit. Passed " + std::to_string(qbits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override {
      return 1; 
    }

    virtual std::string label() const override { 
      return "Rz"; 
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
      return gate;
    }

    GATECLONE(RzRotationGate);
};

template <class GateType>
class MemoizedGate : public GateType {
  private:
    static std::vector<Eigen::MatrixXcd> memoized_gates;
    static bool defined;

    static void generate_memoized_gates(uint32_t res, double min, double max) {
      MemoizedGate<GateType>::memoized_gates = std::vector<Eigen::MatrixXcd>(res);
      double bin_width = (max - min)/res;
      std::vector<uint32_t> qubits{0};

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
    MemoizedGate(const std::vector<uint32_t>& qbits) : GateType(qbits) {}

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
      return MemoizedGate<GateType>::memoized_gates[idx];
    }

    GATECLONE(MemoizedGate<GateType>);
};

template <class GateType>
std::vector<Eigen::MatrixXcd> MemoizedGate<GateType>::memoized_gates;

template <class GateType>
bool MemoizedGate<GateType>::defined = false;

static std::shared_ptr<Gate> parse_gate(const std::string& s, const std::vector<uint32_t>& qbits) {
  Eigen::MatrixXcd data(1u << qbits.size(), 1u << qbits.size());
  std::complex<double> i(0.0, 1.0);
  double sqrt2 = 1.41421356237;

  if (s == "H" || s == "h") {
    data << 1.0, 1.0, 1.0, -1.0;
    data /= sqrt2;
    return std::make_shared<MatrixGate>(data, qbits, "h");
  } else if (s == "X" || s == "x") {
    data << 0.0, 1.0, 1.0, 0.0;
    return std::make_shared<MatrixGate>(data, qbits, "x");
  } else if (s == "Y" || s == "y") {
    data << 0.0, 1.0*i, -1.0*i, 0.0;
    return std::make_shared<MatrixGate>(data, qbits);
  } else if (s == "Z" || s == "z") {
    data << 1.0, 0.0, 0.0, -1.0;
    return std::make_shared<MatrixGate>(data, qbits);
  } else if (s == "RX" || s == "Rx" || s == "rx") {
    return std::make_shared<RxRotationGate>(qbits);
  } else if (s == "RXM" || s == "Rxm" || s == "rxm") {
    return std::make_shared<MemoizedGate<RxRotationGate>>(qbits);
  } else if (s == "RY" || s == "Ry" || s == "ry") {
    return std::make_shared<RyRotationGate>(qbits);
  } else if (s == "RYM" || s == "Rym" || s == "rym") {
    return std::make_shared<MemoizedGate<RyRotationGate>>(qbits);
  } else if (s == "RZ" || s == "Rz" || s == "rz") {
    return std::make_shared<RzRotationGate>(qbits);
  } else if (s == "RZM" || s == "Rzm" || s == "rzm") {
    return std::make_shared<MemoizedGate<RzRotationGate>>(qbits);
  } else if (s == "CZ" || s == "cz") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, -1.0;
    return std::make_shared<MatrixGate>(data, qbits, "cz");
  } else if (s == "CX" || s == "cx") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0;
    return std::make_shared<MatrixGate>(data, qbits, "cx");
  } else if (s == "CY" || s == "cy") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0*i,
            0.0, 0.0, 1.0*i, 0.0;
    return std::make_shared<MatrixGate>(data, qbits, "cy");
  } else if (s == "swap" || s == "SWAP") {
    data << 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
    return std::make_shared<MatrixGate>(data, qbits, "swap");
  } else {
    throw std::invalid_argument("Invalid gate type: " + s);
  }
}
