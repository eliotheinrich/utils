#pragma once

#include "Frame.h"
#include "Tableau.hpp"
#include <deque>
#include <algorithm>
#include <Samplers.h>

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

class CliffordState : public EntropyState {
  public:
    CliffordState()=default;

    CliffordState(uint32_t num_qubits, int seed=-1) : EntropyState(num_qubits) {
      if (seed == -1) {
        thread_local std::random_device rd;
        rng = std::minstd_rand(rd());
      }
      else rng = std::minstd_rand(seed);
    }

    uint32_t rand() {
      return this->rng();
    }

    double randf() {
      return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
    }

    virtual ~CliffordState() {}

    uint32_t system_size() const { 
      return EntropyState::system_size; 
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
      h(b);
      cz(a, b);
      h(b);
    }

    virtual void cy(uint32_t a, uint32_t b) {
      s(b);
      h(b);
      cz(a, b);
      h(b);
      sd(b);
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

      for (uint32_t i = 0; i < system_size(); i++) {
        e += mzr_expectation(i);
      }

      return e/system_size();
    }

    virtual double mxr_expectation(uint32_t a) {
      h(a);
      double p = mzr_expectation(a);
      h(a);
      return p;
    }
    virtual double mxr_expectation() {
      double e = 0.0;

      for (uint32_t i = 0; i < system_size(); i++) {
        e += mxr_expectation(i);
      }

      return e/system_size();
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

      for (uint32_t i = 0; i < system_size(); i++) {
        e += myr_expectation(i);
      }

      return e/system_size();
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

    void random_clifford(std::vector<uint32_t> &qubits) {
      uint32_t num_qubits = qubits.size();
      std::deque<uint32_t> dqubits(num_qubits);
      std::copy(qubits.begin(), qubits.end(), dqubits.begin());
      for (uint32_t i = 0; i < num_qubits; i++) {
        random_clifford_iteration(dqubits);
        dqubits.pop_front();
      }
    }

    virtual std::string to_string() const { return ""; };

    virtual double sparsity() const=0;

  protected:
    std::minstd_rand rng;

  private:
    // Returns the circuit which maps a PauliString to Z1 if z, otherwise to X1
    void single_qubit_random_clifford(uint32_t a, uint32_t r) {
      // r == 0 is identity, so do nothing in thise case
      if (r == 1) {
        x(a);
      } else if (r == 2) {
        y(a);
      } else if (r == 3) {
        z(a);
      } else if (r == 4) {
        h(a);
        s(a);
        h(a);
        s(a);
      } else if (r == 5) {
        h(a);
        s(a);
        h(a);
        s(a);
        x(a);
      } else if (r == 6) {
        h(a);
        s(a);
        h(a);
        s(a);
        y(a);
      } else if (r == 7) {
        h(a);
        s(a);
        h(a);
        s(a);
        z(a);
      } else if (r == 8) {
        h(a);
        s(a);
      } else if (r == 9) {
        h(a);
        s(a);
        x(a);
      } else if (r == 10) {
        h(a);
        s(a);
        y(a);
      } else if (r == 11) {
        h(a);
        s(a);
        z(a);
      } else if (r == 12) {
        h(a);
      } else if (r == 13) {
        h(a);
        x(a);
      } else if (r == 14) {
        h(a);
        y(a);
      } else if (r == 15) {
        h(a);
        z(a);
      } else if (r == 16) {
        s(a);
        h(a);
        s(a);
      } else if (r == 17) {
        s(a);
        h(a);
        s(a);
        x(a);
      } else if (r == 18) {
        s(a);
        h(a);
        s(a);
        y(a);
      } else if (r == 19) {
        s(a);
        h(a);
        s(a);
        z(a);
      } else if (r == 20) {
        s(a);
      } else if (r == 21) {
        s(a);
        x(a);
      } else if (r == 22) {
        s(a);
        y(a);
      } else if (r == 23) {
        s(a);
        z(a);
      }
    }

    // Performs an iteration of the random clifford algorithm outlined in https://arxiv.org/pdf/2008.06011.pdf
    void random_clifford_iteration(std::deque<uint32_t> &qubits) {
      uint32_t num_qubits = qubits.size();

      // If only acting on one qubit, can easily lookup from a table
      if (num_qubits == 1) {
        single_qubit_random_clifford(qubits[0], rand() % 24);
        return;
      }

      PauliString p1 = PauliString::rand(num_qubits, rng);
      PauliString p2 = PauliString::rand(num_qubits, rng);
      while (p1.commutes(p2)) {
        p2 = PauliString::rand(num_qubits, rng);
      }

      tableau_utils::Circuit c1 = p1.reduce(false);

      apply_circuit(c1, p2);

      auto qubit_map_visitor = tableau_utils::overloaded{
        [&qubits](tableau_utils::sgate s) ->  tableau_utils::Gate { return tableau_utils::sgate{qubits[s.q]}; },
          [&qubits](tableau_utils::sdgate s) -> tableau_utils::Gate { return tableau_utils::sdgate{qubits[s.q]}; },
          [&qubits](tableau_utils::hgate s) ->  tableau_utils::Gate { return tableau_utils::hgate{qubits[s.q]}; },
          [&qubits](tableau_utils::cxgate s) -> tableau_utils::Gate { return tableau_utils::cxgate{qubits[s.q1], qubits[s.q2]}; }
      };

      for (auto &gate : c1) {
        gate = std::visit(qubit_map_visitor, gate);
      }

      apply_circuit(c1, *this);

      PauliString z1p = PauliString::basis(num_qubits, "Z", 0, false);
      PauliString z1m = PauliString::basis(num_qubits, "Z", 0, true);

      if (p2 != z1p && p2 != z1m) {
        tableau_utils::Circuit c2 = p2.reduce(true);

        for (auto &gate : c2) {
          gate = std::visit(qubit_map_visitor, gate);
        }

        apply_circuit(c2, *this);
      }
    }
};
