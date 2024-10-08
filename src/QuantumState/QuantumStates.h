
#include "QuantumCircuit.h"
#include "EntropyState.hpp"

#include <map>
#include <bitset>
#include <iostream>

#include <Eigen/Dense>
#include <fmt/format.h>


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

class EntropyState;

using PauliAmplitude = std::pair<PauliString, double>;

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

		QuantumState() = default;

		QuantumState(uint32_t num_qubits, int s=-1) : EntropyState(num_qubits), num_qubits(num_qubits), basis(1u << num_qubits) {
			if (s == -1) {
        thread_local std::random_device gen;
				seed(gen());
			} else {
				seed(s);
			}
		}

		virtual void seed(int s) {
			rng.seed(s);
		}

    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples)=0;

    virtual double stabilizer_renyi_entropy(size_t index, const std::vector<PauliAmplitude>& samples) {
      std::vector<double> amplitude_samples;
      for (const auto &[P, p] : samples) {
        amplitude_samples.push_back(p);
      }

      if (index == 1) {
        double q = 0.0;
        for (size_t i = 0; i < amplitude_samples.size(); i++) {
          double p = amplitude_samples[i];
          q += -std::log(p*p);
        }

        q = q/samples.size();
        return -q;
      } else {
        double q = 0.0;
        for (size_t i = 0; i < amplitude_samples.size(); i++) {
          double p = amplitude_samples[i];
          q += std::pow(p, 2*(index - 1));
        }

        q = q/amplitude_samples.size();
        return 1.0/(1.0 - index) * std::log(q);
      }
    }

    virtual double stabilizer_renyi_entropy(size_t index) {
      // Default to 1000 samples
      std::vector<PauliAmplitude> samples = stabilizer_renyi_entropy_samples(1000);

      return stabilizer_renyi_entropy(index, samples);
    }

    std::vector<PauliAmplitude> stabilizer_renyi_entropy_montecarlo(size_t num_samples);
    std::vector<PauliAmplitude> stabilizer_renyi_entropy_exhaustive();

    virtual std::vector<PauliAmplitude> stabilizer_renyi_entropy_samples(size_t num_samples) {
      return stabilizer_renyi_entropy_montecarlo(num_samples);
    }

    virtual std::complex<double> expectation(const PauliString& p) const=0;

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

    virtual void evolve(const QuantumCircuit& qc, const std::vector<uint32_t>& qubits) {
      if (qubits.size() != qc.num_qubits) {
        throw std::runtime_error("Provided qubits do not match size of circuit.");
      }

      QuantumCircuit qc_mapped(qc);
      qc_mapped.resize(num_qubits);
      qc_mapped.apply_qubit_map(qubits);
      
      evolve(qc_mapped);
    }

    void random_clifford(std::vector<uint32_t> &qubits) {
      QuantumCircuit qc = ::random_clifford(qubits.size(), rng);
      evolve(qc, qubits);
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
class MatrixProductOperator;

class DensityMatrix : public QuantumState {
	public:
		Eigen::MatrixXcd data;

    DensityMatrix()=default;

		DensityMatrix(uint32_t num_qubits);

		DensityMatrix(const Statevector& state);

		DensityMatrix(const QuantumCircuit& circuit);

		DensityMatrix(const DensityMatrix& rho);

    DensityMatrix(const UnitaryState& U);

    DensityMatrix(const MatrixProductState& mps);

		DensityMatrix(const Eigen::MatrixXcd& data);

		virtual std::string to_string() const override;

		DensityMatrix partial_trace(const std::vector<uint32_t>& traced_qubits) const;

    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;

		virtual void evolve(const Eigen::MatrixXcd& gate) override;

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) override;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const std::vector<uint32_t>& qbits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool measure(uint32_t q) override;

		virtual std::vector<bool> measure_all() override;

		Eigen::VectorXd diagonal() const;

		virtual std::vector<double> probabilities() const override;
		std::map<uint32_t, double> probabilities_map() const;
};

class Statevector : public QuantumState {
	public:
		Eigen::VectorXcd data;

    Statevector()=default;

		Statevector(uint32_t num_qubits);

		Statevector(uint32_t num_qubits, uint32_t qregister);

		Statevector(const QuantumCircuit &circuit);

		Statevector(const Statevector& other);

		Statevector(const Eigen::VectorXcd& vec);

		Statevector(const MatrixProductState& state);

		virtual std::string to_string() const override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override;
    virtual double stabilizer_renyi_entropy(size_t index) override {
      return DensityMatrix(*this).stabilizer_renyi_entropy(index);
    }

    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override {
      return DensityMatrix(*this).magic_mutual_information(qubitsA, qubitsB, num_samples);
    }

    virtual std::complex<double> expectation(const PauliString& p) const override {
      return DensityMatrix(*this).expectation(p);
    }

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve(const Eigen::MatrixXcd &gate) override;
		
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void evolve(const QuantumCircuit& circuit, const std::vector<bool>& outcomes);

		double measure_probability(uint32_t q, bool outcome) const;

		virtual bool measure(uint32_t q) override;

    bool measure(uint32_t q, bool outcome);

		double norm() const;

		void normalize();

		void fix_gauge();

		double probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const;

		virtual std::vector<double> probabilities() const override;

		std::map<uint32_t, double> probabilities_map() const;

		std::complex<double> inner(const Statevector& other) const;

		Eigen::VectorXd svd(const std::vector<uint32_t>& qubits) const;
};

class UnitaryState : public QuantumState {
	public:
		Eigen::MatrixXcd unitary;

    UnitaryState()=default;

		UnitaryState(uint32_t num_qubits);

		virtual std::string to_string() const override;

		virtual double entropy(const std::vector<uint32_t> &sites, uint32_t index) override;

    virtual double stabilizer_renyi_entropy(size_t index) override {
      return DensityMatrix(*this).stabilizer_renyi_entropy(index);
    }

    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override {
      return DensityMatrix(*this).magic_mutual_information(qubitsA, qubitsB, num_samples);
    }

    virtual std::complex<double> expectation(const PauliString& p) const override {
      return DensityMatrix(*this).expectation(p);
    }

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve(const Eigen::MatrixXcd &gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool measure(uint32_t q) override {
			throw std::invalid_argument("Cannot perform measurement on UnitaryState.");
		}

		void normalize();

		Statevector get_statevector() const;

		double probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const {
			return get_statevector().probabilities(z, qubits);
		}

		virtual std::vector<double> probabilities() const override {
			return get_statevector().probabilities();
		}
};

class MatrixProductStateImpl;

class MatrixProductState : public QuantumState {
  friend class MatrixProductOperator;

	private:
    std::unique_ptr<MatrixProductStateImpl> impl;

	public:
    MatrixProductState()=default;
    ~MatrixProductState();

		MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold=1e-8);
    MatrixProductState(const MatrixProductState& other);

    static MatrixProductState ising_ground_state(size_t num_qubits, double h, size_t bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);

    virtual void seed(int i) override;
		virtual std::string to_string() const override;

		virtual double entropy(const std::vector<uint32_t>& qubits, uint32_t index) override;

    virtual std::vector<PauliAmplitude> stabilizer_renyi_entropy_samples(size_t num_samples) override;
    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;

		void print_mps() const;

    MatrixProductOperator partial_trace(const std::vector<uint32_t>& qubits) const;

		std::complex<double> coefficients(uint32_t z) const;
		Eigen::VectorXcd coefficients(const std::vector<uint32_t>& indices) const;
		Eigen::VectorXcd coefficients() const;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) override;
		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		double measure_probability(uint32_t q, bool outcome) const;

		virtual std::vector<double> probabilities() const override {
			Statevector statevector(*this);
			return statevector.probabilities();
		}
		virtual bool measure(uint32_t q) override;
};

class MatrixProductOperatorImpl;

class MatrixProductOperator : public QuantumState {
  private:
    std::unique_ptr<MatrixProductOperatorImpl> impl;

  public:
    MatrixProductOperator()=default;
    ~MatrixProductOperator();

    MatrixProductOperator(const MatrixProductState& mps, const std::vector<uint32_t>& traced_qubits);
    void print_mps() const;
		Eigen::MatrixXcd coefficients() const;
    virtual std::complex<double> expectation(const PauliString& p) const override;

    virtual std::vector<double> probabilities() const override {
      return DensityMatrix(coefficients()).probabilities();
    }

    virtual std::string to_string() const override {
      return DensityMatrix(coefficients()).to_string();
    }

    virtual std::vector<double> magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override {
      throw std::runtime_error("magic_mutual_information not implemented for MatrixProductOperator.");
    }

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override {
      throw std::runtime_error("entropy not implemented for MatrixProductOperator.");
    }

    virtual bool measure(uint32_t q) override {
      throw std::runtime_error("measure not implemented for MatrixProductOperator.");
    }

    virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) override {
      throw std::runtime_error("evolve not implemented for MatrixProductOperator.");
    }

};
