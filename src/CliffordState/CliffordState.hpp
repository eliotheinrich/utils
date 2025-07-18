#pragma once

#include "QuantumCircuit.h"
#include "EntanglementEntropyState.hpp"
#include "Random.hpp"

#include <algorithm>

enum CliffordType { CHP, GraphSim };

static inline CliffordType parse_clifford_type(std::string s) {
  if (s == "chp") {
    return CliffordType::CHP;
  } else if (s == "graph") {
    return CliffordType::GraphSim;
  } else {
    throw std::invalid_argument("Cannot parse clifford state type.");
  }
}

class CliffordState : public EntanglementEntropyState {
  public:
    size_t num_qubits;
    CliffordState()=default;

    CliffordState(uint32_t num_qubits) : EntanglementEntropyState(num_qubits), num_qubits(num_qubits) {}
    virtual ~CliffordState() {}

    void evolve(const QuantumCircuit& qc, const Qubits& qubits) {
      if (qubits.size() != qc.get_num_qubits()) {
        throw std::runtime_error("Provided qubits do not match size of circuit.");
      }

      QuantumCircuit qc_mapped(qc);
      qc_mapped.resize(num_qubits);
      qc_mapped.apply_qubit_map(qubits);
      
      evolve(qc_mapped);
    }

    void evolve(const QuantumCircuit& qc) {
      if (!qc.is_clifford()) {
        throw std::runtime_error("Provided circuit is not Clifford.");
      }

			for (auto const &inst : qc.instructions) {
				evolve(inst);
			}
    }

		void evolve(const Instruction& inst) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
          std::string name = gate->label();

          if (name == "H") {
            h(gate->qubits[0]);
          } else if (name == "S") {
            s(gate->qubits[0]);
          } else if (name == "Sd") {
            sd(gate->qubits[0]);
          } else if (name == "CX") {
            cx(gate->qubits[0], gate->qubits[1]);
          } else if (name == "X") {
            x(gate->qubits[0]);
          } else if (name == "Y") {
            y(gate->qubits[0]);
          } else if (name == "Z") {
            z(gate->qubits[0]);
          } else if (name == "CY") {
            cy(gate->qubits[0], gate->qubits[1]);
          } else if (name == "CZ") {
            cz(gate->qubits[0], gate->qubits[1]);
          } else if (name == "SWAP") {
            swap(gate->qubits[0], gate->qubits[1]);
          } else {
            throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to CliffordState.evolve.", name));
          }
				},
				[this](const Measurement& m) { 
          if (!m.is_basis()) {
            throw std::runtime_error("Currently, can only perform measurements in computational basis on Clifford states.");
          }

          mzr(m.qubits[0]);
				},
        [this](const WeakMeasurement& m) {
          throw std::runtime_error("Cannot perform weak measurements on Clifford states.");
        }
			}, inst);
		}

    virtual void h(uint32_t a)=0;

    virtual void s(uint32_t a)=0;

    virtual void sd(uint32_t a) {
      s(a);
      s(a);
      s(a);
    }

    virtual void x(uint32_t a) {
      h(a);
      z(a);
      h(a);
    }

    virtual void y(uint32_t a) {
      x(a);
      z(a);
    }

    virtual void z(uint32_t a) {
      s(a);
      s(a);
    }

    virtual void sqrtx(uint32_t a) {
      h(a);
      s(a);
      h(a);
    }
    
    virtual void sqrty(uint32_t a) {
      h(a);
      s(a);
      sqrtx(a);
      sd(a);
      h(a);
    }

    virtual void sqrtz(uint32_t a) {
      s(a);
    }

    virtual void sqrtxd(uint32_t a) {
      h(a);
      sd(a);
      h(a);
    }

    virtual void sqrtyd(uint32_t a) {
      h(a);
      sd(a);
      sqrtxd(a);
      s(a);
      h(a);
    }

    virtual void sqrtzd(uint32_t a) {
      sd(a);
    }

    virtual void cz(uint32_t a, uint32_t b)=0;

    virtual void cx(uint32_t a, uint32_t b) {
      h(a);
      cz(b, a);
      h(a);
    }

    virtual void cy(uint32_t a, uint32_t b) {
      s(a);
      h(a);
      cz(b, a);
      h(a);
      sd(a);
    }

    virtual void swap(uint32_t a, uint32_t b) {
      cx(a, b);
      cx(b, a);
      cx(a, b);
    }

    virtual void T4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
      cx(a, d);
      cx(b, d);
      cx(c, d);

      cx(d, a);
      cx(d, b);
      cx(d, c);

      cx(a, d);
      cx(b, d);
      cx(c, d);
    }

    virtual double mzr_expectation(uint32_t a)=0;
    virtual double mzr_expectation() {
      double e = 0.0;

      for (uint32_t i = 0; i < num_qubits; i++) {
        e += mzr_expectation(i);
      }

      return e/num_qubits;
    }

    virtual double mxr_expectation(uint32_t a) {
      h(a);
      double p = mzr_expectation(a);
      h(a);
      return p;
    }
    virtual double mxr_expectation() {
      double e = 0.0;

      for (uint32_t i = 0; i < num_qubits; i++) {
        e += mxr_expectation(i);
      }

      return e/num_qubits;
    }

    virtual double myr_expectation(uint32_t a) {
      s(a);
      h(a);
      double p = mzr_expectation(a);
      h(a);
      sd(a);
      return p;
    }
    virtual double myr_expectation() {
      double e = 0.0;

      for (uint32_t i = 0; i < num_qubits; i++) {
        e += myr_expectation(i);
      }

      return e/num_qubits;
    }

    virtual bool mzr(uint32_t a)=0;

    virtual bool mxr(uint32_t a) {
      h(a);
      bool outcome = mzr(a);
      h(a);
      return outcome;
    }

    virtual bool myr(uint32_t a) {
      s(a);
      h(a);
      bool outcome = mzr(a);
      h(a);
      sd(a);
      return outcome;
    }

    virtual void random_clifford(const Qubits& qubits)=0;

    virtual std::string to_string() const=0;

    virtual double sparsity() const=0;
};
