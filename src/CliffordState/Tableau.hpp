#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"
#include "QuantumCircuit.h"

#include <Eigen/Dense>

#include <fmt/format.h>

class Tableau {
  public:
    bool track_destabilizers;
    uint32_t num_qubits;
    std::vector<PauliString> rows;

    Tableau()=default;

    Tableau(uint32_t num_qubits) : track_destabilizers(true), num_qubits(num_qubits) {
      rows = std::vector<PauliString>(2*num_qubits + 1, PauliString(num_qubits));
      for (uint32_t i = 0; i < num_qubits; i++) {
        rows[i].set_x(i, true);
        rows[i + num_qubits].set_z(i, true);
      }
    }

    Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows) : track_destabilizers(false), num_qubits(num_qubits), rows(rows) {}

    uint32_t num_rows() const { 
      if (track_destabilizers) { 
        return rows.size() - 1; 
      } else {
        return rows.size(); 
      }
    }

    Eigen::MatrixXi to_matrix() const {
      Eigen::MatrixXi M(num_qubits, 2*num_qubits);
      size_t offset = track_destabilizers ? num_qubits : 0;
      
      for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
          M(i, j) = get_z(i + offset, j);
          M(i, j + num_qubits) = get_x(i + offset, j);
        }
      }

      return M;
    }

    Statevector to_statevector() const {
      if (num_qubits > 15) {
        throw std::runtime_error("Cannot create a Statevector with more than 31 qubits.");
      }

      Eigen::MatrixXcd dm = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);
      Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);

      for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
        PauliString p = rows[i];
        Eigen::MatrixXcd g = p.to_matrix();
        dm = dm*((I + g)/2.0);
      }

      uint32_t N = 1u << num_qubits;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(dm);
      Eigen::VectorXcd vec = solver.eigenvectors().block(0,N-1,N,1).rowwise().reverse();

      return Statevector(vec);
    }

    bool operator==(Tableau& other) {
      if (num_qubits != other.num_qubits) {
        return false;
      }

      rref();
      other.rref();

      int32_t r1 = track_destabilizers ? num_qubits : 0;
      for (uint32_t i = r1; i < num_rows(); i++) {
        if (get_r(i) != other.get_r(i)) {
          return false;
        }

        for (uint32_t j = 0; j < num_qubits; j++) {
          if (get_z(i, j) != other.get_z(i, j)) {
            return false;
          }

          if (get_x(i, j) != other.get_x(i, j)) {
            return false;
          }
        }
      }

      return true;
    }

    // Put tableau into reduced row echelon form
    void rref(const Qubits& sites) {
      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t pivot_row = 0;
      uint32_t row = r1;

      for (uint32_t k = 0; k < 2*sites.size(); k++) {
        uint32_t c = sites[k % sites.size()];
        bool z = k < sites.size();
        bool found_pivot = false;
        for (uint32_t i = row; i < r2; i++) {
          if ((z && rows[i].get_z(c)) || (!z && rows[i].get_x(c))) {
            pivot_row = i;
            found_pivot = true;
            break;
          }
        }

        if (found_pivot) {
          std::swap(rows[row], rows[pivot_row]);

          for (uint32_t i = r1; i < r2; i++) {
            if (i == row) {
              continue;
            }

            if ((z && rows[i].get_z(c)) || (!z && rows[i].get_x(c))) {
              rowsum(i, row);
            }
          }

          row += 1;
        } else {
          continue;
        }
      }
    }

    void xrref(const Qubits& sites) {
      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t pivot_row = 0;
      uint32_t row = r1;

      for (uint32_t k = 0; k < 2*sites.size(); k++) {
        uint32_t c = sites[k % sites.size()];
        bool z = k < sites.size();
        bool found_pivot = false;
        for (uint32_t i = row; i < r2; i++) {
          if (!z && rows[i].get_x(c)) {
            pivot_row = i;
            found_pivot = true;
            break;
          }
        }

        if (found_pivot) {
          std::swap(rows[row], rows[pivot_row]);

          for (uint32_t i = r1; i < r2; i++) {
            if (i == row) {
              continue;
            }

            if (!z && rows[i].get_x(c)) {
              rowsum(i, row);
            }
          }

          row += 1;
        } else {
          continue;
        }
      }
    }

    uint32_t xrank(const Qubits& sites) {
      xrref(sites);

      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t r = 0;
      for (uint32_t i = r1; i < r2; i++) {
        for (uint32_t j = 0; j < sites.size(); j++) {
          if (rows[i].get_x(sites[j])) {
            r++;
            break;
          }
        }
      }

      return r;
    }

    uint32_t rank(const Qubits& sites) {
      rref(sites);

      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t r = 0;
      for (uint32_t i = r1; i < r2; i++) {
        for (uint32_t j = 0; j < sites.size(); j++) {
          if (rows[i].get_x(sites[j]) || rows[i].get_z(sites[j])) {
            r++;
            break;
          }
        }
      }

      return r;
    }

    void rref() {
      std::vector<uint32_t> qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      rref(qubits);
    }

    void xrref() {
      std::vector<uint32_t> qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      xrref(qubits);
    }

    uint32_t rank() {
      std::vector<uint32_t> qubits(num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      return rank(qubits);
    }

    inline void validate_qubit(uint32_t a) const {
      if (!(a >= 0 && a < num_qubits)) {
        std::string error_message = "A gate was applied to qubit " + std::to_string(a) + 
          ", which is outside of the allowed range (0, " + std::to_string(num_qubits) + ").";
        throw std::invalid_argument(error_message);
      }
    }

    std::string to_string(bool print_destabilizers=true) const {
      std::string s = "";
      uint32_t i1 = print_destabilizers ? 0 : num_qubits;
      for (uint32_t i = i1; i < num_rows(); i++) {
        s += (i == i1) ? "[" : " ";
        s += rows[i].to_string();
        s += (i == num_rows() - 1) ? "]" : "\n";
      }
      return s;
    }

    std::string to_string_ops(bool print_destabilizers=true) const {
      std::string s = "";
      uint32_t i1 = print_destabilizers ? 0 : num_qubits;
      for (uint32_t i = i1; i < num_rows(); i++) {
        s += (i == i1) ? "[" : " ";
        s += "[" + rows[i].to_string_ops() + "]";
        s += (i == num_rows() - 1) ? "]" : "\n";
      }
      return s + "]";
    }

    void rowsum(uint32_t h, uint32_t i) {
      rows[h] = rows[h] * rows[i];
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
          } else if (name == "X") {
            x(gate->qubits[0]);
          } else if (name == "Y") {
            y(gate->qubits[0]);
          } else if (name == "Z") {
            z(gate->qubits[0]);
          } else if (name == "Sd") {
            sd(gate->qubits[0]);
          } else if (name == "CX") {
            cx(gate->qubits[0], gate->qubits[1]);
          } else if (name == "CZ") {
            cz(gate->qubits[0], gate->qubits[1]);
          } else {
            throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to Tableau.evolve.", name));
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
				},
			}, inst);
		}

    void h(uint32_t a) {
      validate_qubit(a);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].h(a);
      }
    }

    void s(uint32_t a) {
      validate_qubit(a);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].s(a);
      }
    }

    void sd(uint32_t a) {
      s(a);
      s(a);
      s(a);
    }

    void x(uint32_t a) {
      validate_qubit(a);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].x(a);
      }
    }

    void y(uint32_t a) {
      validate_qubit(a);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].y(a);
      }
    }

    void z(uint32_t a) {
      validate_qubit(a);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].z(a);
      }
    }

    void cx(uint32_t a, uint32_t b) {
      validate_qubit(a);
      validate_qubit(b);
      for (uint32_t i = 0; i < num_rows(); i++) {
        rows[i].cx(a, b);
      }
    }

    void cz(uint32_t a, uint32_t b) {
      h(b);
      cx(a, b);
      h(b);
    }

    // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
    // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
    std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) const {
      if (!track_destabilizers) {
        throw std::invalid_argument("Cannot check mzr_deterministic without track_destabilizers.");
      }

      for (uint32_t p = num_qubits; p < 2*num_qubits; p++) {
        // Suitable p identified; outcome is random
        if (get_x(p, a)) { 
          return std::pair(false, p);
        }
      }

      // No p found; outcome is deterministic
      return std::pair(true, 0);
    }

    bool mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) {
      validate_qubit(a);
      if (!track_destabilizers) {
        throw std::invalid_argument("Cannot mzr without track_destabilizers.");
      }


      auto [deterministic, p] = mzr_deterministic(a);

      if (!deterministic) {
        bool b;
        if (outcome) {
          b = outcome.value();
        } else {
          b = randi() % 2;
        }

        for (uint32_t i = 0; i < 2*num_qubits; i++) {
          if (i != p && get_x(i, a)) {
            rowsum(i, p);
          }
        }

        std::swap(rows[p - num_qubits], rows[p]);
        rows[p] = PauliString(num_qubits);

        set_r(p, b);
        set_z(p, a, true);

        return b;
      } else { // deterministic
        rows[2*num_qubits] = PauliString(num_qubits);
        for (uint32_t i = 0; i < num_qubits; i++) {
          if (rows[i].get_x(a)) {
            rowsum(2*num_qubits, i + num_qubits);
          }
        }

        bool b = get_r(2*num_qubits);
        if (outcome) {
          if (b != outcome.value()) {
            throw std::runtime_error("Invalid forced measurement of QuantumCHPState.");
          }
        }

        return b;
      }
    }

    double sparsity() const {
      float nonzero = 0;
      for (uint32_t i = 0; i < num_rows(); i++) {
        for (uint32_t j = 0; j < num_qubits; j++) {
          nonzero += rows[i].get_x(j);
          nonzero += rows[i].get_z(j);
        }
      }

      return nonzero/(num_rows()*num_qubits*2);
    }


    inline bool get_x(uint32_t i, uint32_t j) const { 
      return rows[i].get_x(j); 
    }

    inline bool get_z(uint32_t i, uint32_t j) const { 
      return rows[i].get_z(j); 
    }

    inline bool get_r(uint32_t i) const { 
      uint8_t r = rows[i].get_r();
      if (r == 0) {
        return false;
      } else if (r == 2) {
        return true;
      } else {
        throw std::runtime_error("Anomolous phase detected in Clifford tableau.");
      }
    }

    inline void set_x(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_x(j, v); 
    }

    inline void set_z(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_z(j, v); 
    }

    inline void set_r(uint32_t i, bool v) { 
      if (v) {
        rows[i].set_r(2);
      } else {
        rows[i].set_r(0);

      }
    }
};

#include <glaze/glaze.hpp>

template<>
struct glz::meta<Tableau> {
  static constexpr auto value = glz::object(
    "num_qubits", &Tableau::num_qubits,
    "track_destabilizers", &Tableau::track_destabilizers,
    "rows", &Tableau::rows
  );
};
