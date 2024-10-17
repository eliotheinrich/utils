#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"
#include "QuantumCircuit.h"

#include <Eigen/Dense>

class Tableau {
  public:
    bool track_destabilizers;
    uint32_t num_qubits;
    std::vector<PauliString> rows;

    Tableau()=default;

    Tableau(uint32_t num_qubits) :
      track_destabilizers(true), num_qubits(num_qubits) {
        rows = std::vector<PauliString>(2*num_qubits + 1, PauliString(num_qubits));
        for (uint32_t i = 0; i < num_qubits; i++) {
          rows[i].set_x(i, true);
          rows[i + num_qubits].set_z(i, true);
        }
      }

    Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows)
      : track_destabilizers(false), num_qubits(num_qubits), rows(rows) {}

    uint32_t num_rows() const { 
      if (track_destabilizers) { 
        return rows.size() - 1; 
      } else {
        return rows.size(); 
      }
    }

    //Eigen::MatrixXi to_matrix() const {
    //  Eigen::MatrixXi M(num_rows(), 2*num_rows());
    //  for (size_t i = 0; i < num_rows(); i++) {
    //    for (size_t j = 0; j < 2*num_rows(); j++) {
    //      M(i, j) = 
    //    }
    //  }
    //}

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
        if (r(i) != other.r(i)) {
          return false;
        }

        for (uint32_t j = 0; j < num_qubits; j++) {
          if (z(i, j) != other.z(i, j)) {
            return false;
          }

          if (x(i, j) != other.x(i, j)) {
            return false;
          }
        }
      }

      return true;
    }

    // Put tableau into reduced row echelon form
    void rref(const std::vector<uint32_t>& sites) {
      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t pivot_row = 0;
      uint32_t row = r1;

      for (uint32_t k = 0; k < 2*sites.size(); k++) {
        uint32_t c = sites[k % sites.size()];
        bool z = k < sites.size();
        bool found_pivot = false;
        for (uint32_t i = row; i < r2; i++) {
          if ((z && rows[i].z(c)) || (!z && rows[i].x(c))) {
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

            if ((z && rows[i].z(c)) || (!z && rows[i].x(c))) {
              rowsum(i, row);
            }
          }

          row += 1;
        } else {
          continue;
        }
      }
    }

    uint32_t rank(const std::vector<uint32_t>& sites) {
      rref(sites);

      uint32_t r1 = track_destabilizers ? num_qubits : 0;
      uint32_t r2 = num_rows();

      uint32_t r = 0;
      for (uint32_t i = r1; i < r2; i++) {
        for (uint32_t j = 0; j < sites.size(); j++) {
          if (rows[i].x(sites[j]) || rows[i].z(sites[j])) {
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

    std::string to_string() const {
      std::string s = "";
      for (uint32_t i = 0; i < num_rows(); i++) {
        s += (i == 0) ? "[" : " ";
        s += rows[i].to_string();
        s += (i == num_rows() - 1) ? "]" : "\n";
      }
      return s;
    }

    std::string to_string_ops() const {
      std::string s = "";
      for (uint32_t i = 0; i < num_rows(); i++) {
        s += (i == 0) ? "[" : " ";
        s += "[" + rows[i].to_string_ops() + "]";
        s += (i == num_rows() - 1) ? "]" : "\n";
      }
      return s + "]";
    }

    void rowsum(uint32_t h, uint32_t i) {
      rows[h] *= rows[i];
    }

    void evolve(const QuantumCircuit& qc, std::minstd_rand& rng) {
      if (!qc.is_clifford()) {
        throw std::runtime_error("Provided circuit is not Clifford.");
      }

			for (auto const &inst : qc.instructions) {
				evolve(inst, rng);
			}
    }

		void evolve(const Instruction& inst, std::minstd_rand& rng) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
          std::string name = gate->label();

          if (name == "H") {
            h(gate->qbits[0]);
          } else if (name == "S") {
            s(gate->qbits[0]);
          } else if (name == "Sd") {
            sd(gate->qbits[0]);
          } else if (name == "CX") {
            cx(gate->qbits[0], gate->qbits[1]);
          } else if (name == "CZ") {
            cz(gate->qbits[0], gate->qbits[1]);
          } else {
            throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to Tableau.evolve.", name));
          }
				},
				[this, &rng](Measurement m) { 
					for (auto const &q : m.qbits) {
						mzr(q, rng);
					}
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
    std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) {
      if (!track_destabilizers) {
        throw std::invalid_argument("Cannot check mzr_deterministic without track_destabilizers.");
      }

      for (uint32_t p = num_qubits; p < 2*num_qubits; p++) {
        // Suitable p identified; outcome is random
        if (x(p, a)) { 
          return std::pair(false, p);
        }
      }

      // No p found; outcome is deterministic
      return std::pair(true, 0);
    }

    bool mzr(uint32_t a, std::minstd_rand& rng) {
      validate_qubit(a);
      if (!track_destabilizers) {
        throw std::invalid_argument("Cannot mzr without track_destabilizers.");
      }


      auto [deterministic, p] = mzr_deterministic(a);

      if (!deterministic) {
        bool outcome = rng() % 2;
        for (uint32_t i = 0; i < 2*num_qubits; i++) {
          if (i != p && x(i, a)) {
            rowsum(i, p);
          }
        }

        std::swap(rows[p - num_qubits], rows[p]);
        rows[p] = PauliString(num_qubits);

        set_r(p, outcome);
        set_z(p, a, true);

        return outcome;
      } else { // deterministic
        rows[2*num_qubits] = PauliString(num_qubits);
        for (uint32_t i = 0; i < num_qubits; i++) {
          rowsum(2*num_qubits, i + num_qubits);
        }

        return r(2*num_qubits);
      }
    }

    double sparsity() const {
      float nonzero = 0;
      for (uint32_t i = 0; i < num_rows(); i++) {
        for (uint32_t j = 0; j < num_qubits; j++) {
          nonzero += rows[i].x(j);
          nonzero += rows[i].z(j);
        }
      }

      return nonzero/(num_rows()*num_qubits*2);
    }


    inline bool x(uint32_t i, uint32_t j) const { 
      return rows[i].x(j); 
    }

    inline bool z(uint32_t i, uint32_t j) const { 
      return rows[i].z(j); 
    }

    inline bool r(uint32_t i) const { 
      return rows[i].r(); 
    }

    inline void set_x(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_x(j, v); 
    }

    inline void set_z(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_z(j, v); 
    }

    inline void set_r(uint32_t i, bool v) { 
      rows[i].set_r(v); 
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

template<>
struct glz::meta<PauliString> {
  static constexpr auto value = glz::object(
    "num_qubits", &PauliString::num_qubits,
    "phase", &PauliString::phase,
    "width", &PauliString::width,
    "bit_string", &PauliString::bit_string
  );
};
