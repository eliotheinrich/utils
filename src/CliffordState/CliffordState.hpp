#pragma once

#include "QuantumState.h"
#include "QuantumCircuit.h"
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

class CliffordState : public QuantumState {
  public:
    size_t num_qubits;
    CliffordState()=default;

    CliffordState(uint32_t num_qubits) : QuantumState(num_qubits), num_qubits(num_qubits) {}
    virtual ~CliffordState() {}

    virtual void evolve(const QuantumCircuit& qc, const Qubits& qubits) override {
      if (qubits.size() != qc.get_num_qubits()) {
        throw std::runtime_error("Provided qubits do not match size of circuit.");
      }

      QuantumCircuit qc_mapped(qc);
      qc_mapped.resize(num_qubits);
      qc_mapped.apply_qubit_map(qubits);
      
      evolve(qc_mapped);
    }

    virtual void evolve(const QuantumCircuit& qc) override {
      if (!qc.is_clifford()) {
        throw std::runtime_error("Provided circuit is not Clifford.");
      }

			for (auto const &inst : qc.instructions) {
				evolve(inst);
			}
    }

		virtual void evolve(const Instruction& inst) override {
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
          measure(m);
				},
        [this](const WeakMeasurement& m) {
          throw std::runtime_error("Cannot perform weak measurements on Clifford states.");
        }
			}, inst);
		}

    virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override {
      throw std::runtime_error("Cannot evolve arbitrary gate on Clifford state.");
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

    virtual double mzr_expectation(uint32_t a) const=0;

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

    virtual bool mzr(uint32_t a, std::optional<bool> outcome=std::nullopt)=0;

    virtual bool mxr(uint32_t a, std::optional<bool> outcome=std::nullopt) {
      h(a);
      bool b = mzr(a, outcome);
      h(a);
      return b;
    }

    virtual bool myr(uint32_t a, std::optional<bool> outcome=std::nullopt) {
      s(a);
      h(a);
      bool b = mzr(a, outcome);
      h(a);
      sd(a);
      return b;
    }

    virtual std::string to_string() const=0;

    virtual double sparsity() const=0;

    virtual bool measure(const Measurement& m) override {
      if (m.is_basis()) {
        return mzr(m.qubits[0]);
      } else {
        QuantumCircuit qc(m.qubits.size());

        auto args = argsort(m.qubits);
        m.pauli.value().reduce(true, std::make_pair(&qc, args));

        uint32_t q = std::ranges::min(m.qubits);

        evolve(qc, m.qubits);
        bool outcome = mzr(q, m.outcome);
        evolve(qc.adjoint(), m.qubits);

        return outcome;
      }
    }

    virtual bool weak_measure(const WeakMeasurement& m) override {
      throw std::runtime_error("Cannot call a weak measurement on a Clifford state.");
    }

    // TODO there is a bug in the sign of expectations. Fix it.
    virtual std::complex<double> expectation(const PauliString& pauli) const {
      QuantumCircuit qc(num_qubits);
      Qubits qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      uint32_t q = std::ranges::min(qubits);

      pauli.reduce(true, std::make_pair(&qc, qubits));

      auto self = const_cast<CliffordState*>(this);
      self->evolve(qc);
      double exp = self->mzr_expectation(q);
      self->evolve(qc.adjoint());

      return pauli.sign() * std::complex<double>(exp, 0.0);
    }

    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override {
      throw std::runtime_error("Not yet implemented!");
    }

    // TEST THIS
    virtual std::vector<BitAmplitudes> sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
      if (num_qubits > 15) {
        throw std::runtime_error("Cannot sample bitstrings for Clifford state with n > 15 qubits.");
      }

      std::vector<double> probs = probabilities();
      auto marginal_probs = marginal_probabilities(supports);

      std::discrete_distribution<> dist(probs.begin(), probs.end()); 
      std::minstd_rand rng(randi());

      std::vector<BitAmplitudes> samples;

      for (size_t i = 0; i < num_samples; i++) {
        size_t z = dist(rng);
        BitString bits = BitString::from_bits(z, num_qubits);
        std::vector<double> amplitudes = {probs[z]};
        for (size_t n = 0; n < supports.size(); n++) {
          amplitudes[n-1] = marginal_probs[n][z];
        }

        samples.push_back({bits, amplitudes});
      }

      return samples;
    }

		virtual std::vector<double> probabilities() const {
      if (num_qubits > 15) {
        throw std::runtime_error("Cannot generate probabilities for Clifford state with n > 15 qubits.");
      }

      size_t b = 1u << num_qubits;
      std::vector<double> probs(b);
      for (size_t i = 0; i < b; i++) {
        double p = 1.0;
        for (size_t q = 0; q < num_qubits; q++) {
          if (std::abs(mzr_expectation(q)) < 1e-5) {
            p *= 0.5;
          }
        }
        probs[i] = p;
      }

      return probs;
    }

    virtual double purity() const override {
      return 1.0;
    }

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override {
      throw std::runtime_error("Cannot evaluate partial_trace on Clifford states.");
    }
};
