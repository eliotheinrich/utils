#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <variant>
#include <string>
#include <complex>
#include <memory>

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

    Eigen::MatrixXcd adjoint(const std::vector<double>& params) const {
      return define(params).adjoint();
    }

    Eigen::MatrixXcd adjoint() const {
      if (num_params() > 0) {
        throw std::invalid_argument("Unbound parameters; cannot define gate.");
      }

      return adjoint(std::vector<double>());
    }

    virtual std::shared_ptr<Gate> clone()=0;
};

class MatrixGate : public Gate {
  public:
    Eigen::MatrixXcd data;
    std::string label_str;

    MatrixGate(const Eigen::MatrixXcd& data, const std::vector<uint32_t>& qbits, const std::string& label_str)
      : Gate(qbits), data(data), label_str(label_str) {}

    MatrixGate(const Eigen::MatrixXcd& data, const std::vector<uint32_t>& qbits)
      : MatrixGate(data, qbits, "U") {}


    uint32_t num_params() const override {
      return 0;
    }

    std::string label() const override {
      return label_str;
    }

    Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (params.size() != 0) {
        throw std::invalid_argument("Cannot pass parameters to MatrixGate.");
      }

      return data;
    }

    std::shared_ptr<Gate> clone() override { 
      return std::shared_ptr<Gate>(new MatrixGate(data, qbits)); 
    }
};

struct Measurement {
  std::vector<uint32_t> qbits;
  Measurement(const std::vector<uint32_t>& qbits) : qbits(qbits) {}
};

typedef std::variant<std::shared_ptr<Gate>, Measurement> Instruction;
