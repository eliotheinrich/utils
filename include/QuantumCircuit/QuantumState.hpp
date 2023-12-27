#pragma once

#include "QuantumCircuit.h"

#include <EntropySampler.hpp>

#include <map>
#include <bitset>

#include <Eigen/Dense>
#include <itensor/all.h>

#include <iostream>

#define QS_ATOL 1e-8

namespace quantumstate_utils {
	static inline bool print_congruence(uint32_t z1, uint32_t z2, const std::vector<uint32_t>& pos, bool outcome) {
		std::bitset<8> b1(z1);
		std::bitset<8> b2(z2);
		if (outcome) {
			std::cout << b1 << " and " << b2 << " are congruent at positions ";
		} else {
			std::cout << b1 << " and " << b2 << " are not congruent at positions ";
		}

		for (auto p : pos) {
			std::cout << p << " ";
		}
		std::cout << "\n";

		return outcome;
	}

	static inline bool bits_congruent(uint32_t z1, uint32_t z2, const std::vector<uint32_t>& pos) {
		for (uint32_t j = 0; j < pos.size(); j++) {
			if (((z2 >> j) & 1) != ((z1 >> pos[j]) & 1)) {
				return false;
			}
		}

		return true;
	}

	static inline uint32_t reduce_bits(uint32_t a, const std::vector<uint32_t>& v) {
		uint32_t b = 0;

		for (size_t i = 0; i < v.size(); i++) {
			// Get the ith bit of a
			int a_bit = (a >> v[i]) & 1;

			// Set the ith bit of b based on a_bit
			b |= (a_bit << i);
		}

		return b;
	}

	inline uint32_t set_bit(uint32_t b, uint32_t j, uint32_t a, uint32_t i) {
		uint32_t x = (a >> i) & 1u;
		return (b & ~(1u << j)) | (x << j);
	}


	static inline std::string print_binary(uint32_t a, uint32_t width=5) {
		std::string s = "";
		for (uint32_t i = 0; i < width; i++) {
			s = std::to_string((a >> i) & 1u) + s;
		}

		return s;
	}
}


class QuantumState : public EntropyState {
	protected:
    std::minstd_rand rng;

    uint32_t rand() { 
      return this->rng(); 
    }

    double randf() { 
      return double(rand())/double(RAND_MAX); 
    }

	public:
		uint32_t num_qubits;
		uint32_t basis;

		virtual ~QuantumState() = default;
		QuantumState() = default;
		QuantumState(uint32_t num_qubits, int s=-1) : EntropyState(num_qubits), num_qubits(num_qubits), basis(1u << num_qubits) {
			if (s == -1) {
				seed(std::rand());
			} else {
				seed(s);
			}
		}

		void seed(int s) {
			rng.seed(s);
		}

		virtual std::string to_string() const=0;

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits)=0;

		virtual void evolve(const Eigen::MatrixXcd& gate) {
			std::vector<uint32_t> qbits(num_qubits);
			std::iota(qbits.begin(), qbits.end(), 0);
			evolve(gate, qbits);
		}

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t q) {
			std::vector<uint32_t> qbit{q};
			evolve(gate, qbit); 
		}

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const std::vector<uint32_t>& qbits) { 
			evolve(Eigen::MatrixXcd(gate.asDiagonal()), qbits); 
		}

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) { 
			evolve(Eigen::MatrixXcd(gate.asDiagonal())); 
		}

		virtual void evolve(const Measurement& measurement) {
			for (auto q : measurement.qbits) {
				measure(q);
			}
		}

		virtual void evolve(const Instruction& inst) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
					evolve(gate->define(), gate->qbits); 
				},
				[this](Measurement m) { 
					for (auto const &q : m.qbits) {
						measure(q);
					}
				},
			}, inst);
		}

		virtual void evolve(const QuantumCircuit& circuit) {
			if (circuit.num_params() > 0) {
				throw std::invalid_argument("Unbound QuantumCircuit parameters; cannot evolve StateVector.");
			}

			for (auto const &inst : circuit.instructions) {
				evolve(inst);
			}
		}

		virtual bool measure(uint32_t q)=0;
		
		virtual std::vector<bool> measure_all() {
			std::vector<bool> outcomes(num_qubits);
			for (uint32_t q = 0; q < num_qubits; q++) {
				outcomes[q] = measure(q);
			}
			return outcomes;
		}

		virtual std::vector<double> probabilities() const=0;
};

class DensityMatrix;
class Statevector;
class UnitaryState;
class MatrixProductState;

class DensityMatrix : public QuantumState {
	public:
		Eigen::MatrixXcd data;

    DensityMatrix()=default;
		
    DensityMatrix(uint32_t num_qubits) : QuantumState(num_qubits) {
      data = Eigen::MatrixXcd::Zero(basis, basis);
      data(0, 0) = 1;
    }

		DensityMatrix(const Statevector& state) : QuantumState(state.num_qubits) {
      data = Eigen::kroneckerProduct(state.data.adjoint(), state.data);
    }

		DensityMatrix(const QuantumCircuit& circuit) : DensityMatrix(circuit.num_qubits) {
      evolve(circuit);
    }

		DensityMatrix(const DensityMatrix& rho) : DensityMatrix(rho.num_qubits) {
      data = rho.data;
    }

    DensityMatrix(const Eigen::MatrixXcd& data) : DensityMatrix(std::log2(data.rows())), data(data) {
    }

    virtual std::string to_string() const override {
      std::stringstream ss;
      ss << data;
      return "DensityMatrix(" + std::to_string(num_qubits) + "):\n" + ss.str() + "\n";
    }

    DensityMatrix partial_trace(const std::vector<uint32_t>& traced_qubits) const {
      uint32_t num_qubits = std::log2(data.rows());

      std::vector<uint32_t> remaining_qubits;
      for (uint32_t q = 0; q < num_qubits; q++) {
        if (!std::count(traced_qubits.begin(), traced_qubits.end(), q)) {
          remaining_qubits.push_back(q);
        }
      }

      uint32_t num_traced_qubits = traced_qubits.size();
      uint32_t num_remaining_qubits = remaining_qubits.size();

      DensityMatrix reduced_rho(num_remaining_qubits);
      reduced_rho.data = Eigen::MatrixXcd::Zero(1u << num_remaining_qubits, 1u << num_remaining_qubits);

      uint32_t s = 1u << num_traced_qubits;
      uint32_t h = 1u << num_remaining_qubits;

      for (uint32_t r = 0; r < h; r++) {
        for (uint32_t c = 0; c < h; c++) {
          // Indices into full rho
          uint32_t i = 0;
          uint32_t j = 0;
          for (uint32_t k = 0; k < num_remaining_qubits; k++) {
            // Set bits in (i,j) corresponding to remaining qubits
            uint32_t q = remaining_qubits[k];
            i = quantumstate_utils::set_bit(i, q, r, k);
            j = quantumstate_utils::set_bit(j, q, c, k);
          }

          for (uint32_t n = 0; n < s; n++) {
            for (uint32_t k = 0; k < num_traced_qubits; k++) {
              // Set bits in (i,j) corresponding to traced qubits
              uint32_t q = traced_qubits[k];
              i = quantumstate_utils::set_bit(i, q, n, k);
              j = quantumstate_utils::set_bit(j, q, n, k);
            }

            reduced_rho.data(r, c) += data(i, j);
          }
        }
      }

      return reduced_rho;
    }

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override {
      // If number of qubits is larger than half the system, take advantage of the fact that 
      // S_A = S_\bar{A} to compute entropy for the smaller of A and \bar{A}
      if (qubits.size() > num_qubits) {
        std::vector<uint32_t> qubits_complement;
        for (uint32_t q = 0; q < num_qubits; q++) {
          if (!std::count(qubits.begin(), qubits.end(), q)) {
            qubits_complement.push_back(q);
          }
        }

        return entropy(qubits_complement, index);
      }



      std::vector<uint32_t> traced_qubits;
      for (uint32_t q = 0; q < num_qubits; q++) {
        if (!std::count(qubits.begin(), qubits.end(), q)) {
          traced_qubits.push_back(q);
        }
      }


      DensityMatrix rho_a = partial_trace(traced_qubits);

      if (index == 0) {
        uint32_t rank = 0;
        // TODO cleanup
        for (auto const &e : rho_a.data.eigenvalues()) {
          if (std::abs(e) > QS_ATOL) {
            rank++;
          }
        }
        return std::log2(rank);
      } else if (index == 1) {
        double s = 0.;
        for (auto const &e : rho_a.data.eigenvalues()) {
          double eigenvalue = std::abs(e);
          if (eigenvalue > QS_ATOL) {
            s -= std::abs(e)*std::log(std::abs(e));
          }
        }
        return s;
      } else {
        return 1./(1. - index) * std::log(rho_a.data.pow(index).trace().real());
      }
    }

		virtual void evolve(const Eigen::MatrixXcd& gate) override {
      data = gate * data * gate.adjoint();
    }

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) override {
      evolve(full_circuit_unitary(gate, qbits, num_qubits));
    }

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const std::vector<uint32_t>& qbits) override {
      uint32_t s = 1u << num_qubits;
      uint32_t h = 1u << qbits.size();

      if (gate.size() != h) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      for (uint32_t a1 = 0; a1 < s; a1++) {
        for (uint32_t a2 = 0; a2 < s; a2++) {
          uint32_t b1 = quantumstate_utils::reduce_bits(a1, qbits);
          uint32_t b2 = quantumstate_utils::reduce_bits(a2, qbits);

          data(a1, a2) *= gate(b1)*std::conj(gate(b2));
        }
      }
    }

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override {
      uint32_t s = 1u << num_qubits;

      if (gate.size() != s) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      for (uint32_t a1 = 0; a1 < s; a1++) {
        for (uint32_t a2 = 0; a2 < s; a2++) {
          data(a1, a2) *= gate(a1)*std::conj(gate(a2));
        }
      }
    }

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

	  virtual bool measure(uint32_t q) override {
      for (uint32_t i = 0; i < basis; i++) {
        for (uint32_t j = 0; j < basis; j++) {
          uint32_t q1 = (i >> q) & 1u;
          uint32_t q2 = (j >> q) & 1u;

          if (q1 != q2) {
            data(i, j) = 0;
          }
        }
      }

      return 0;
    }


		virtual std::vector<bool> measure_all() override {
      for (uint32_t i = 0; i < basis; i++) {
        for (uint32_t j = 0; j < basis; j++) {
          if (i != j) {
            data(i, j) = 0;
          }
        }
      }

      return std::vector<bool>(num_qubits, 0);
    }

		Eigen::VectorXd diagonal() const {
      return data.diagonal().cwiseAbs();
    }

		virtual std::vector<double> probabilities() const override {
      std::vector<double> probs(basis);
      for (uint32_t i = 0; i < basis; i++) {
        probs[i] = data(i, i).real();
      }

      return probs;
    }

		std::map<uint32_t, double> probabilities_map() const {
      std::map<uint32_t, double> outcomes;

      for (uint32_t i = 0; i < basis; i++) {
        outcomes[i] = data(i, i).real();
      }

      return outcomes;
    }
};

class Statevector : public QuantumState {
	public:
		Eigen::VectorXcd data;

    Statevector()=default;

		Statevector(uint32_t num_qubits, uint32_t qregister) {
      data = Eigen::VectorXcd::Zero(1u << num_qubits);
      data(qregister) = 1.;
    }

		Statevector(uint32_t num_qubits) : Statevector(num_qubits, 0) {}

		Statevector(const QuantumCircuit &circuit) : Statevector(circuit.num_qubits, 0) {
      evolve(circuit);
    }

    Statevector(const Eigen::VectorXcd& vec) : Statevector(std::log2(vec.size())), data(vec) {
      uint32_t s = vec.size();
      if ((s & (s - 1)) != 0) {
        throw std::invalid_argument("Provided data to Statevector does not have a dimension which is a power of 2.");
      }
    }

		Statevector(const Statevector& other) : Statevector(other.data) {}

		Statevector(const MatrixProductState& state) : Statevector(state.coefficients()) {}

		virtual std::string to_string() const override {
      Statevector tmp(*this);
      tmp.fix_gauge();

      uint32_t s = 1u << tmp.num_qubits;

      bool first = true;
      std::string st = "";
      for (uint32_t i = 0; i < s; i++) {
        if (std::abs(tmp.data(i)) > QS_ATOL) {
          std::string amplitude;
          if (std::abs(tmp.data(i).imag()) < QS_ATOL) {
            amplitude = std::to_string(tmp.data(i).real());
          } else {
            amplitude = "(" + std::to_string(tmp.data(i).real()) + ", " + std::to_string(tmp.data(i).imag()) + ")";
          }

          std::string bin = quantumstate_utils::print_binary(i, tmp.num_qubits);

          if (!first) {
            st += " + ";
          }
          first = false;
          st += amplitude + "|" + bin + ">";
        }
      }

      return st;
    }

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override {
      DensityMatrix rho(*this);
      return rho.entropy(qubits, index);
    }

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) override {
      uint32_t s = 1u << num_qubits;
      uint32_t h = 1u << qubits.size();
      if ((gate.rows() != h) || gate.cols() != h) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      Eigen::VectorXcd ndata = Eigen::VectorXcd::Zero(s);

      for (uint32_t a1 = 0; a1 < s; a1++) {
        uint32_t b1 = quantumstate_utils::reduce_bits(a1, qubits);

        for (uint32_t b2 = 0; b2 < h; b2++) {
          uint32_t a2 = a1;
          for (uint32_t j = 0; j < qubits.size(); j++) {
            a2 = quantumstate_utils::set_bit(a2, qubits[j], b2, j);
          }

          ndata(a1) += gate(b1, b2)*data(a2);
        }
      }

      data = ndata;
    }

		virtual void evolve(const Eigen::MatrixXcd &gate) override {
      if (!(gate.rows() == data.size() && gate.cols() == data.size())) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      data = gate*data;
    }

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate, const std::vector<uint32_t> &qubits) override {
      uint32_t s = 1u << num_qubits;
      uint32_t h = 1u << qubits.size();

      if (gate.size() != h) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      for (uint32_t a = 0; a < s; a++) {
        uint32_t b = quantumstate_utils::reduce_bits(a, qubits);

        data(a) *= gate(b);
      }
    }

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate) override {
      uint32_t s = 1u << num_qubits;

      if (gate.size() != s) {
        throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
      }

      for (uint32_t a = 0; a < s; a++) {
        data(a) *= gate(a);
      }
    }

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void evolve(const QuantumCircuit& circuit, const std::vector<bool>& outcomes) {
      uint32_t d = 0;
      for (auto const &inst : circuit.instructions) {
        if (inst.index() == 0) {
          QuantumState::evolve(inst);
        } else {
          Measurement m = std::get<Measurement>(inst);
          for (auto const& q : m.qbits) {
            measure(q, outcomes[d]);
            d++;
          }
        }
      }
    }

		double measure_probability(uint32_t q, bool outcome) const {
      uint32_t s = 1u << num_qubits;

      double prob_zero = 0.0;
      for (uint32_t i = 0; i < s; i++) {
        if (((i >> q) & 1u) == 0) {
          prob_zero += std::pow(std::abs(data(i)), 2);
        }
      }

      if (outcome) {
        return 1.0 - prob_zero;
      } else {
        return prob_zero;
      }
    }

    bool measure(uint32_t q, bool outcome) {
      uint32_t s = 1u << num_qubits;
      for (uint32_t i = 0; i < s; i++) {
        if (((i >> q) & 1u) != outcome) {
          data(i) = 0.;
        }
      }

      normalize();

      return outcome;
    }

		virtual bool measure(uint32_t q) override {
      uint32_t s = 1u << num_qubits;

      double prob_zero = measure_probability(q, 0);
      uint32_t outcome = !(randf() < prob_zero);

      for (uint32_t i = 0; i < s; i++) {
        if (((i >> q) & 1u) != outcome) {
          data(i) = 0.;
        }
      }

      normalize();

      return outcome;
    }


		double norm() const {
      double n = 0.;
      for (uint32_t i = 0; i < data.size(); i++) {
        n += std::pow(std::abs(data(i)), 2);
      }

      return std::sqrt(n);
    }

		void normalize() {
    data = data/norm();
    }

		void fix_gauge() {
      uint32_t j = 0;
      uint32_t s = 1u << num_qubits;
      for (uint32_t i = 0; i < s; i++) {
        if (std::abs(data(i)) > QS_ATOL) {
          j = i;
          break;
        }
      }

      std::complex<double> a = data(j)/std::abs(data(j));

      data = data/a;
    }

		double probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const {
      uint32_t s = 1u << num_qubits;
      double p = 0.;
      for (uint32_t i = 0; i < s; i++) {
        if (quantumstate_utils::bits_congruent(i, z, qubits)) {
          p += std::pow(std::abs(data(i)), 2);
        }
      }

      return p;
    }

		virtual std::vector<double> probabilities() const override {
      uint32_t s = 1u << num_qubits;

      std::vector<double> probs(s);
      for (uint32_t i = 0; i < s; i++) {
        probs[i] = std::pow(std::abs(data(i)), 2);
      }

      return probs;
    }

    std::map<uint32_t, double> probabilities_map() const {
      std::vector<double> probs = probabilities();


      std::map<uint32_t, double> probs_map;
      for (uint32_t i = 0; i < probs.size(); i++) {
        probs_map.emplace(i, probs[i]);
      }

      return probs_map; 
    }

		std::complex<double> inner(const Statevector& other) const {
      uint32_t s = 1u << num_qubits;

      std::complex<double> c = 0.;
      for (uint32_t i = 0; i < s; i++) {
        c += other.data(i)*std::conj(data(i));
      }

      return c;
    }

		Eigen::VectorXd svd(const std::vector<uint32_t>& qubits) const {
      Eigen::MatrixXcd matrix(data);

      uint32_t r = 1u << qubits.size();
      uint32_t c = 1u << (num_qubits - qubits.size());
      matrix.resize(r, c);

      Eigen::JacobiSVD<Eigen::MatrixXcd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
      return svd.singularValues();
    }
};

class UnitaryState : public QuantumState {
	public:
		Eigen::MatrixXcd unitary;

    UnitaryState()=default;
		UnitaryState(uint32_t num_qubits) : QuantumState(num_qubits) {
      unitary = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
      unitary.setIdentity();
    }

		virtual std::string to_string() const override {
      return get_statevector().to_string();
    }

		virtual double entropy(const std::vector<uint32_t> &sites, uint32_t index) override {
      return get_statevector().entropy(sites, index);
    }

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) {
      evolve(full_circuit_unitary(gate, qubits, num_qubits));
    }

		virtual void evolve(const Eigen::MatrixXcd &gate) override {
      unitary = gate * unitary;
    }

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool measure(uint32_t q) {
			throw std::invalid_argument("Cannot perform measurement on UnitaryState.");
		}

		void normalize() {
      unitary = normalize_unitary(unitary);
    }

		Statevector get_statevector() const {
      Statevector statevector(num_qubits);
      statevector.evolve(unitary);
      return statevector;
    }

		double probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const {
			return get_statevector().probabilities(z, qubits);
		}

		virtual std::vector<double> probabilities() const override {
			return get_statevector().probabilities();
		}
};


itensor::ITensor tensor_slice(const itensor::ITensor& tensor, const itensor::Index& index, int i) {
	if (!hasIndex(tensor, index)) {
		throw std::invalid_argument("Provided tensor cannot be sliced by provided index.");
	}

	auto v = itensor::ITensor(index);
	v.set(i, 1.0);

	return tensor*v;
}

itensor::ITensor matrix_to_tensor(
  const Eigen::Matrix2cd& matrix, 
  const itensor::Index i1, 
  const itensor::Index i2
) {

  itensor::ITensor tensor(i1, i2);

	for (uint32_t i = 1; i <= 2; i++) {
		for (uint32_t j = 1; j <= 2; j++) {
			tensor.set(i1=i, i2=j, matrix(i-1,j-1));
		}
	}


	return tensor;
}

itensor::ITensor matrix_to_tensor(
  const Eigen::Matrix4cd& matrix, 
  const itensor::Index i1, 
  const itensor::Index i2, 
  const itensor::Index i3,
  const itensor::Index i4
) {

  itensor::ITensor tensor(i1, i2, i3, i4);

	for (uint32_t i = 1; i <= 2; i++) {
		for (uint32_t j = 1; j <= 2; j++) {
			for (uint32_t k = 1; k <= 2; k++) {
				for (uint32_t l = 1; l <= 2; l++) {
					tensor.set(i1=i, i2=j, i3=k, i4=l, matrix(2*(j-1) + (i-1), 2*(l-1) + (k-1)));
				}
			}
		}
	}

	return tensor;
}

itensor::Index pad(itensor::ITensor& tensor, const itensor::Index& idx, uint32_t new_dim) {
	if (!hasIndex(tensor, idx)) {
		throw std::invalid_argument("Provided tensor does not have provided index.");
	}

	uint32_t old_dim = dim(idx);
	if (old_dim > new_dim) {
		throw std::invalid_argument("Provided dimension is smaller than existing dimension.");
	}

	if (old_dim == new_dim) {
		return idx;
	}

  itensor::Index new_idx(new_dim, idx.tags());

	std::vector<Index> new_inds;
	uint32_t j = -1;
	auto old_inds = inds(tensor);
	for (uint32_t i = 0; i < old_inds.size(); i++) {
		if (old_inds[i] == idx) {
			j = i;
		}

		new_inds.push_back(old_inds[i]);
	}

	new_inds[j] = new_idx;
  itensor::ITensor new_tensor(new_inds);

	for (const auto& it : itensor::iterInds(new_tensor)) {
		if (it[j].val <= old_dim) {
			std::vector<uint32_t> idx_vals(it.size());
			for (uint32_t j = 0; j < it.size(); j++) {
				idx_vals[j] = it[j].val;
			}

			new_tensor.set(it, eltC(tensor, idx_vals));
		}
	}
	
	tensor = new_tensor;

	return new_idx;
}

class MatrixProductState : public QuantumState {
	private:
		std::vector<itensor::ITensor> tensors;
		std::vector<itensor::ITensor> singular_values;
		std::vector<itensor::Index> external_indices;
		std::vector<itensor::Index> internal_indices;

		double sv_threshold;

		static Eigen::Matrix2cd zero_projector() {
      Eigen::Matrix2cd P;
      P << 1, 0, 0, 0;
      return P;
    }

		static Eigen::Matrix2cd one_projector() {
      Eigen::Matrix2cd P;
      P << 0, 0, 0, 1;
      return P;
    }

	public:
		uint32_t bond_dimension;

    MatrixProductState()=default;

    MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold=1e-4)
    : QuantumState(num_qubits), sv_threshold(sv_threshold), bond_dimension(bond_dimension) {
      if (bond_dimension > 1u << num_qubits) {
        throw std::invalid_argument("Bond dimension must be smaller than 2^num_qubits.");
      }

      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        internal_indices.push_back(Index(1, "Internal,Left,a" + std::to_string(i)));
        internal_indices.push_back(Index(1, "Internal,Right,a" + std::to_string(i)));
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        external_indices.push_back(Index(2, "External,i" + std::to_string(i)));
      }

      itensor::ITensor tensor;

      tensor = itensor::ITensor(internal_indices[0], external_indices[0]);
      for (auto j : range1(internal_indices[0])) {
        tensor.set(j, 1, 1.0);
      }
      tensors.push_back(tensor);

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        tensor = ITensor(internal_indices[2*i], internal_indices[2*i + 1]);
        tensor.set(1, 1, 1);
        singular_values.push_back(tensor);

        if (i == num_qubits - 2) {
          tensor = itensor::ITensor(internal_indices[2*i + 1], external_indices[i+1]);
          for (auto j : range1(internal_indices[2*i + 1])) {
            tensor.set(j, 1, 1.0);
          }
        } else {
          tensor = itensor::ITensor(internal_indices[2*i + 1], internal_indices[2*i + 2], external_indices[i+1]);
          for (auto j1 : range1(internal_indices[2*i + 1])) {
            tensor.set(j1, j1, 1, 1.0);
          }
        }

        tensors.push_back(tensor);
      }
    }

    virtual std::string to_string() const override {
      Statevector state(*this);
      return state.to_string();
    }

    virtual double entropy(const std::vector<uint32_t>& qubits, uint32_t index) override {
      if (index != 1) {
        throw std::invalid_argument("Can only compute von Neumann (index = 1) entropy for MPS states.");
      }

      if (qubits.size() == 0) {
        return 0.0;
      }

      std::vector<uint32_t> sorted_qubits(qubits);
      std::sort(sorted_qubits.begin(), sorted_qubits.end());

      if (sorted_qubits[0] != 0) {
        throw std::invalid_argument("Invalid qubits passed to MatrixProductState.entropy; must be a continuous interval with left side qubit = 0.");
      }

      for (uint32_t i = 0; i < qubits.size() - 1; i++) {
        if (std::abs(int(sorted_qubits[i]) - int(sorted_qubits[i+1])) > 1) {
          throw std::invalid_argument("Invalid qubits passed to MatrixProductState.entropy; must be a continuous interval with left side qubit = 0.");
        }
      }

      uint32_t q = sorted_qubits.back();

      return entropy(q);
    }

    double entropy(uint32_t q) {
      if (q < 0 || q > num_qubits) {
        throw std::invalid_argument("Invalid qubit passed to MatrixProductState.entropy; must have 0 <= q <= num_qubits.");
      }

      if (q == 0 || q == num_qubits) {
        return 0.0;
      }

      auto sv = singular_values[q-1];
      int d = itensor::dim(itensor::inds(sv)[0]);

      double s = 0.0;
      for (int i = 1; i <= d; i++) {
        double v = std::pow(itensor::elt(sv, i, i), 2);
        if (v >= 1e-6) {
          s -= v * std::log(v);
        }
      }

      return s;
    }

    void print_mps() const {
      itensor::print(tensors[0]);
      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        itensor::print(singular_values[i]);
        itensor::print(tensors[i+1]);
      }
    }

    itensor::ITensor coefficient_tensor() const {
      itensor::ITensor C = tensors[0];

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        C *= singular_values[i]*tensors[i+1];
      }

      return C;
    }

    std::complex<double> coefficients(uint32_t z) const {
      auto C = coefficient_tensor();

      std::vector<int> assignments(num_qubits);
      for (uint32_t j = 0; j < num_qubits; j++) {
        assignments[j] = ((z >> j) & 1u) + 1;
      }

      return itensor::eltC(C, assignments);
    }

    Eigen::VectorXcd coefficients(const std::vector<uint32_t>& indices) const {
      auto C = coefficient_tensor();

      Eigen::VectorXcd vals(1u << num_qubits);
      for (uint32_t i = 0; i < indices.size(); i++) {
        uint32_t z = indices[i];
        std::vector<int> assignments(num_qubits);
        for (uint32_t j = 0; j < num_qubits; j++) {
          assignments[j] = ((z >> j) & 1u) + 1;
        }

        vals[i] = itensor::eltC(C, assignments);
      }

      return vals;
    }

    Eigen::VectorXcd coefficients() const {
      std::vector<uint32_t> indices(1u << num_qubits);
      std::iota(indices.begin(), indices.end(), 0);

      return coefficients(indices);
    }

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override {
      auto i = external_indices[qubit];
      auto ip = itensor::prime(i);
      itensor::ITensor tensor = matrix_to_tensor(gate, i, ip);
      tensors[qubit] = itensor::noPrime(tensors[qubit]*tensor);
    }

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) override {
      if (qubits.size() == 1) {
        evolve(gate, qubits[0]);
        return;
      }

      if (qubits.size() != 2) {
        throw std::invalid_argument("Can only evolve two-qubit gates in MPS simulation.");
      }

      uint32_t q1 = std::min(qubits[0], qubits[1]);
      uint32_t q2 = std::max(qubits[0], qubits[1]);

      if (q1 == q2) {
        throw std::invalid_argument("Can only evolve gates on adjacent qubits (for now).");
      }

      auto i1 = external_indices[qubits[0]];
      auto i2 = external_indices[qubits[1]];
      ITensor gate_tensor = matrix_to_tensor(gate, 
          itensor::prime(i1), itensor::prime(i2), 
          i1, i2
          );

      ITensor theta = itensor::noPrime(gate_tensor*tensors[q1]*singular_values[q1]*tensors[q2]);

      std::vector<Index> u_inds{external_indices[q1]};
      std::vector<Index> v_inds{external_indices[q2]};

      if (q1 != 0) {
        auto alpha = internal_indices[2*q1-2];
        u_inds.push_back(alpha);
        theta *= singular_values[q1-1];
      }

      if (q2 != num_qubits - 1) {
        auto gamma = internal_indices[2*q2+1];
        v_inds.push_back(gamma);
        theta *= singular_values[q2];
      }

      auto [U, D, V] = itensor::svd(theta, u_inds, v_inds, 
          {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
          "LeftTags=","Internal,Left,a" + std::to_string(q1),
          "RightTags=","Internal,Right,a" + std::to_string(q1)});

      internal_indices[2*q1] = itensor::commonIndex(U, D);
      internal_indices[2*q2 - 1] = itensor::commonIndex(V, D);

      auto inv = [](Real r) { return 1.0/r; };
      if (q1 != 0) {
        U *= itensor::apply(singular_values[q1-1], inv);
      }
      if (q2 != num_qubits - 1) {
        V *= itensor::apply(singular_values[q2], inv);
      }

      tensors[q1] = U;
      tensors[q2] = V;
      singular_values[q1] = D;

    }

    virtual void evolve(const QuantumCircuit& circuit) override { 
      QuantumState::evolve(circuit); 
    }

    double measure_probability(uint32_t q, bool outcome) const {
      int i = static_cast<int>(outcome) + 1;
      auto idx = external_indices[q];
      auto qtensor = tensor_slice(tensors[q], idx, i);

      if (q > 0) {
        qtensor *= singular_values[q-1];
      }
      if (q < num_qubits - 1) {
        qtensor *= singular_values[q];
      }

      return itensor::real(itensor::sumelsC(qtensor * itensor::dag(qtensor)));
    }

    virtual std::vector<double> probabilities() const override {
      Statevector statevector(*this);
      return statevector.probabilities();
    }

		virtual bool measure(uint32_t q) override {
      double prob_zero = measure_probability(q, 0);
      bool outcome = randf() < prob_zero;

      Eigen::Matrix2cd proj = outcome ? MatrixProductState::one_projector()/std::sqrt(1.0 - prob_zero) :
        MatrixProductState::zero_projector()/std::sqrt(prob_zero);

      evolve(proj, q);

      Eigen::Matrix4cd id;
      id.setIdentity();

      // Propagate right
      for (uint32_t i = q; i < num_qubits - 1; i++) {
        if (itensor::dim(itensor::inds(singular_values[i])[0]) == 1) {
          break;
        }

        evolve(id, {i, i+1});
      }

      // Propagate left
      for (uint32_t i = q; i > 1; i--) {
        if (itensor::dim(itensor::inds(singular_values[i-1])[0]) == 1) {
          break;
        }

        evolve(id, {i-1, i});
      }

      return outcome;
    }

		void measure_propagate(uint32_t q, const Eigen::Matrix2cd& proj) {
      evolve(proj, q);

      // Propagate right
      if ((q < num_qubits - 1) && (itensor::dim(itensor::inds(singular_values[q])[0]) > 1)) {
        measure_propagate(q+1, proj);
      }

      // Propagate left
      if ((q > 0) && (itensor::dim(itensor::inds(singular_values[q-1])[0]) > 1)) {
        measure_propagate(q-1, proj);
      }
    }
};
