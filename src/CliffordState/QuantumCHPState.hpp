#pragma once

#include "CliffordState.hpp"
#include "Tableau.hpp"

class QuantumCHPState : public CliffordState {
  private:
    static uint32_t get_num_qubits(const std::string &s) {
      auto substrings = dataframe::utils::split(s, "\n");
      return substrings.size()/2;
    }


  public:
    Tableau tableau;
    CliffordTable rcs;

    QuantumCHPState()=default;

    QuantumCHPState(uint32_t num_qubits, int seed=-1)
      : CliffordState(num_qubits, seed), tableau(Tableau(num_qubits)) {
    }

    QuantumCHPState(const std::string &s) : CliffordState(get_num_qubits(s)) {
      auto substrings = dataframe::utils::split(s, "\n");

      tableau = Tableau(num_qubits);

      for (uint32_t i = 0; i < substrings.size()-1; i++) {
        substrings[i] = substrings[i].substr(1, substrings[i].size() - 3);
        auto chars = dataframe::utils::split(substrings[i], " | ");

        auto row = chars[0];
        bool r = chars[1][0] == '1';

        for (uint32_t j = 0; j < num_qubits; j++) {
          tableau.set_x(i, j, row[j] == '1');
          tableau.set_z(i, j, row[j + num_qubits] == '1');
        }

        tableau.set_r(i, r);
      }
    }

    bool operator==(QuantumCHPState& other) {
      return tableau == other.tableau;
    }

    bool operator!=(QuantumCHPState& other) {
      return !(tableau == other.tableau);
    }

    virtual std::string to_string() const override {
      return tableau.to_string();
    }

    std::string to_string_ops() const {
      return tableau.to_string_ops();
    }

    Statevector to_statevector() const {
      return tableau.to_statevector();
    }

    virtual void h(uint32_t a) override {
      tableau.h(a);
    }

    virtual void s(uint32_t a) override {
      tableau.s(a);
    }

    virtual void sd(uint32_t a) override {
      tableau.s(a);
      tableau.s(a);
      tableau.s(a);
    }

    virtual void cx(uint32_t a, uint32_t b) override {
      tableau.cx(b, a);
    }

    virtual void cy(uint32_t a, uint32_t b) override {
      tableau.s(a);
      tableau.h(a);
      tableau.cz(b, a);
      tableau.h(a);
      tableau.s(a);
      tableau.s(a);
      tableau.s(a);
    }

    virtual void cz(uint32_t a, uint32_t b) override {
      tableau.h(a);
      tableau.cx(b, a);
      tableau.h(a);
    }

    virtual void random_clifford(std::vector<uint32_t> &qubits) override {
      //QuantumCircuit qc(num_qubits);
      //random_clifford_impl(qubits, rng, *this, qc);
      //std::cout << "produced qc = " << qc.to_string() << "\n";
      QuantumCircuit qc(num_qubits);
      QuantumCircuit rc(qubits.size());
      rcs.apply_random(rng, rc);

      qc.append(rc, qubits);
      qc.apply(*this);
    }

    virtual double mzr_expectation(uint32_t a) override {
      auto [deterministic, _] = tableau.mzr_deterministic(a);
      if (deterministic) {
        return 2*int(mzr(a)) - 1.0;
      } else {
        return 0.0;
      }
    }

    virtual bool mzr(uint32_t a) override {
      return tableau.mzr(a, rng);
    }

    virtual double sparsity() const override {
      return tableau.sparsity();
    }

    virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override {
      uint32_t system_size = this->num_qubits;
      uint32_t partition_size = qubits.size();

      // Optimization; if partition size is larger than half the system size, 
      // compute the entropy for the smaller subsystem
      if (partition_size > system_size / 2) {
        std::vector<uint32_t> qubits_complement;
        for (uint32_t q = 0; q < system_size; q++) {
          if (std::find(qubits.begin(), qubits.end(), q) == qubits.end()) {
            qubits_complement.push_back(q);
          }
        }

        return entropy(qubits_complement, index);
      }

      int rank = tableau.rank(qubits);

      int s = rank - partition_size;

      return static_cast<double>(s);
    }

    int partial_rank(const std::vector<uint32_t> &qubits) {
      return tableau.rank(qubits);
    }

    void set_x(size_t i, size_t j, bool v) {
      tableau.set_x(i, j, v);
    }

    void set_z(size_t i, size_t j, bool v) {
      tableau.set_z(i, j, v);
    }
};

#include <glaze/glaze.hpp>

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    "tableau", &QuantumCHPState::tableau
  );
};
