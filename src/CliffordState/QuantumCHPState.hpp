#pragma once

#include <Simulator.hpp>
#include <QuantumState.h>

#include "CliffordState.hpp"
#include "Tableau.hpp"

#include <glaze/glaze.hpp>

class QuantumCHPState : public CliffordState {
  private:
    static uint32_t get_num_qubits(const std::string &s) {
      auto substrings = dataframe::utils::split(s, "\n");
      return substrings.size()/2;
    }

  public:
    Tableau tableau;
    int print_mode;

    QuantumCHPState()=default;

    QuantumCHPState(uint32_t num_qubits)
      : CliffordState(num_qubits), tableau(Tableau(num_qubits)) {
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
      if (print_mode == 0) {
        return tableau.to_string(true);
      } else if (print_mode == 1) {
        return tableau.to_string(false);
      } else if (print_mode == 2) {
        return tableau.to_string_ops(true);
      } else {
        return tableau.to_string_ops(false);
      }
    }

    void rref() {
      tableau.rref();
    }

    void xrref() {
      tableau.xrref();
    }

    void set_print_mode(const std::string& mode) {
      if (mode == "binary_all") {
        print_mode = 0;
      } else if (mode == "binary") {
        print_mode = 1;
      } else if (mode == "paulis_all") {
        print_mode = 2;
      } else if (mode == "paulis") {
        print_mode = 3;
      } else {
        throw std::runtime_error(fmt::format("Invalid print mode: {}", mode));
      }
    }

    void rowsum(uint32_t q1, uint32_t q2) {
      tableau.rowsum(q1, q2);
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
      tableau.cx(a, b);
    }

    virtual void cy(uint32_t a, uint32_t b) override {
      tableau.s(b);
      tableau.h(b);
      tableau.cz(a, b);
      tableau.h(b);
      tableau.sd(b);
    }

    virtual void cz(uint32_t a, uint32_t b) override {
      tableau.h(b);
      tableau.cx(a, b);
      tableau.h(b);
    }

    PauliString get_row(size_t i) const {
      return tableau.rows[i];
    }

    std::vector<PauliString> stabilizers() const {
      std::vector<PauliString> stabs(tableau.rows.begin() + num_qubits, tableau.rows.end() - 1);
      return stabs;
    }

    size_t size() const {
      return tableau.rows.size() - 1;
    }

    virtual void random_clifford(const Qubits& qubits) override {
      random_clifford_impl(qubits, *this);
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
      return tableau.mzr(a);
    }

    virtual double sparsity() const override {
      return tableau.sparsity();
    }

    virtual double entanglement(const QubitSupport& support, uint32_t index) override {
      auto qubits = to_qubits(support);
      uint32_t system_size = this->num_qubits;
      uint32_t partition_size = qubits.size();

      // Optimization; if partition size is larger than half the system size, 
      // compute the entanglement for the smaller subsystem
      if (partition_size > system_size / 2) {
        std::vector<uint32_t> qubits_complement;
        for (uint32_t q = 0; q < system_size; q++) {
          if (std::find(qubits.begin(), qubits.end(), q) == qubits.end()) {
            qubits_complement.push_back(q);
          }
        }

        return entanglement(qubits_complement, index);
      }

      int rank = tableau.rank(qubits);

      int s = rank - partition_size;

      return static_cast<double>(s);
    }

    int xrank() {
      Qubits qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      return tableau.xrank(qubits);
    }

    int partial_xrank(const Qubits& qubits) {
      return tableau.xrank(qubits);
    }

    int rank() {
      Qubits qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      return tableau.rank(qubits);
    }

    int partial_rank(const Qubits& qubits) {
      return tableau.rank(qubits);
    }

    void set_x(size_t i, size_t j, bool v) {
      tableau.set_x(i, j, v);
    }

    void set_z(size_t i, size_t j, bool v) {
      tableau.set_z(i, j, v);
    }

    double participation_entropy() const {
      //int s = 0;
      //for (size_t i = 0; i < num_qubits; i++) {
      //  PauliString p(num_qubits);
      //  p.set_z(i, 1);
      //  if 

      //}
    }

    std::vector<dataframe::byte_t> serialize() const;

    void deserialize(const std::vector<dataframe::byte_t>& bytes);

    Texture get_texture(Color x_color, Color z_color, Color y_color) {
      Texture texture(num_qubits, num_qubits);

      for (size_t r = 0; r < num_qubits; r++) {
        for (size_t i = 0; i < num_qubits; i++) {
          bool zi = tableau.get_z(r + num_qubits, i);
          bool xi = tableau.get_x(r + num_qubits, i);
          if (zi && xi) {
            texture.set(r, i, y_color);
          } else if (zi) {
            texture.set(r, i, z_color);
          } else if (xi) {
            texture.set(r, i, x_color);
          }
        }
      }

      return texture;
    }
};

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    "tableau", &QuantumCHPState::tableau
  );
};
