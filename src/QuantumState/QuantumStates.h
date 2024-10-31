#include "QuantumCircuit.h"
#include "EntropyState.hpp"

#include <map>
#include <bitset>
#include <iostream>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>


#define QS_ATOL 1e-8

namespace quantumstate_utils {
  constexpr double sqrt2i_ = 0.707106781186547524400844362104849;
  constexpr std::complex<double> i_ = std::complex<double>(0.0, 1.0);

  struct H { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << sqrt2i_, sqrt2i_, sqrt2i_, -sqrt2i_).finished(); };

  struct X { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished(); };
  struct Y { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, -i_, i_, 0.0).finished(); };
  struct Z { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, -1.0).finished(); };

  struct sqrtX { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (1.0 - i_)/2.0, (1.0 - i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtY { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (-1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtZ { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, i_).finished(); };

  struct sqrtXd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtYd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 - i_)/2.0, (-1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtZd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, -i_).finished(); };

  struct T { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, sqrt2i_*(1.0 + i_)).finished(); };
  struct Td { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, sqrt2i_*(1.0 - i_)).finished(); };

  struct CX { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, 1, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, 1, 0, 0).finished(); };
  struct CY { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, -i_, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, i_, 0, 0).finished(); };
  struct CZ { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1).finished(); };
  struct SWAP { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished(); };

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
using PauliMutationFunc = std::function<void(PauliString&, std::minstd_rand&)>;
using ProbabilityFunc = std::function<double(double)>;
using magic_t = double; //std::tuple<double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>;

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

    virtual magic_t magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
      return magic_mutual_information_montecarlo(qubitsA, qubitsB, num_samples, 1000, std::nullopt);
    }
    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt)=0;
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB)=0;
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples)=0;

    static double stabilizer_renyi_entropy(size_t index, const std::vector<PauliAmplitude>& samples) {
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
      std::vector<PauliAmplitude> samples = sample_paulis(1000);
      return QuantumState::stabilizer_renyi_entropy(index, samples);
    }

    std::vector<PauliAmplitude> sample_paulis_montecarlo(size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    std::vector<PauliAmplitude> sample_paulis_exhaustive();
    std::vector<PauliAmplitude> sample_paulis_exact(size_t num_samples, ProbabilityFunc prob);

    virtual std::vector<PauliAmplitude> sample_paulis(size_t num_samples) {
      ProbabilityFunc prob = [](double t) -> double { return std::pow(t, 2.0); };
      return sample_paulis_montecarlo(num_samples, 5*num_qubits, prob);
    }

    virtual std::complex<double> expectation(const PauliString& p) const=0;

		virtual std::string to_string() const=0;

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits)=0;

    template <typename G>
    void evolve_one_qubit_gate(uint32_t q) {
      evolve(G::value, q);
    }

    #define DEFINE_ONE_QUBIT_GATE(name, struct)             \
    void name(uint32_t q) {                                 \
      evolve_one_qubit_gate<quantumstate_utils::struct>(q); \
    }

    #define DEFINE_ALL_ONE_QUBIT_GATES     \
    DEFINE_ONE_QUBIT_GATE(h, H);           \
    DEFINE_ONE_QUBIT_GATE(x, X);           \
    DEFINE_ONE_QUBIT_GATE(y, Y);           \
    DEFINE_ONE_QUBIT_GATE(z, Z);           \
    DEFINE_ONE_QUBIT_GATE(sqrtX, sqrtX);   \
    DEFINE_ONE_QUBIT_GATE(sqrtY, sqrtY);   \
    DEFINE_ONE_QUBIT_GATE(sqrtZ, sqrtZ);   \
    DEFINE_ONE_QUBIT_GATE(sqrtXd, sqrtXd); \
    DEFINE_ONE_QUBIT_GATE(sqrtYd, sqrtYd); \
    DEFINE_ONE_QUBIT_GATE(sqrtZd, sqrtZd); \
    DEFINE_ONE_QUBIT_GATE(s, sqrtZ);       \
    DEFINE_ONE_QUBIT_GATE(sd, sqrtZd);     \

    template <typename G>
    void evolve_two_qubit_gate(uint32_t q1, uint32_t q2) { 
      evolve(G::value, {q1, q2});
    }

    #define DEFINE_TWO_QUBIT_GATE(name, struct)                  \
    void name(uint32_t q1, uint32_t q2) {                        \
      evolve_two_qubit_gate<quantumstate_utils::struct>(q1, q2); \
    }

    #define DEFINE_ALL_TWO_QUBIT_GATES       \
    DEFINE_TWO_QUBIT_GATE(cx, CX)     \
    DEFINE_TWO_QUBIT_GATE(cy, CY)     \
    DEFINE_TWO_QUBIT_GATE(cz, CZ)     \
    DEFINE_TWO_QUBIT_GATE(swap, SWAP)

    DEFINE_ALL_ONE_QUBIT_GATES
    DEFINE_ALL_TWO_QUBIT_GATES

    void random_clifford(const std::vector<uint32_t> &qubits) {
      random_clifford_impl(qubits, rng, *this);
    }

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
				mzr(q);
			}
		}

		virtual void evolve(const Instruction& inst) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
					evolve(gate->define(), gate->qbits); 
				},
				[this](Measurement m) { 
					for (auto const &q : m.qbits) {
						mzr(q);
					}
				},
			}, inst);
		}

		virtual void evolve(const QuantumCircuit& circuit) {
			if (circuit.num_params() > 0) {
				throw std::invalid_argument("Unbound QuantumCircuit parameters; cannot evolve Statevector.");
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

    //void random_clifford(std::vector<uint32_t> &qubits) {
    //  QuantumCircuit qc = ::random_clifford(qubits.size(), rng);
    //  evolve(qc, qubits);
    //}

		virtual bool mzr(uint32_t q)=0;
		
		virtual std::vector<bool> mzr_all() {
			std::vector<bool> outcomes(num_qubits);
			for (uint32_t q = 0; q < num_qubits; q++) {
				outcomes[q] = mzr(q);
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

    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) override;
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) const;

		virtual void evolve(const Eigen::MatrixXcd& gate) override;

		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) override;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const std::vector<uint32_t>& qbits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool mzr(uint32_t q) override;

		virtual std::vector<bool> mzr_all() override;

    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits);

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

    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override {
      return DensityMatrix(*this).magic_mutual_information_montecarlo(qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
    }
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) override {
      return DensityMatrix(*this).magic_mutual_information_exhaustive(qubitsA, qubitsB);
    }
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override {
      return DensityMatrix(*this).magic_mutual_information_exact(qubitsA, qubitsB, num_samples);
    }

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const;

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve(const Eigen::MatrixXcd &gate) override;
		
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void evolve(const QuantumCircuit& circuit, const std::vector<bool>& outcomes);

		double mzr_prob(uint32_t q, bool outcome) const;

		virtual bool mzr(uint32_t q) override;

    bool mzr(uint32_t q, bool outcome);

    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits);
    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta);

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

    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override {
      return DensityMatrix(*this).magic_mutual_information_montecarlo(qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
    }
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) override {
      return DensityMatrix(*this).magic_mutual_information_exhaustive(qubitsA, qubitsB);
    }
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override {
      return DensityMatrix(*this).magic_mutual_information_exact(qubitsA, qubitsB, num_samples);
    }

    virtual std::complex<double> expectation(const PauliString& p) const override {
      return DensityMatrix(*this).expectation(p);
    }

		virtual void evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) override;

		virtual void evolve(const Eigen::MatrixXcd &gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool mzr(uint32_t q) override {
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

    virtual std::vector<PauliAmplitude> sample_paulis(size_t num_samples) override;

    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) override;
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const;

		void print_mps(bool print_data) const;

    MatrixProductOperator partial_trace(const std::vector<uint32_t>& qubits) const;

		std::complex<double> coefficients(uint32_t z) const;
		Eigen::VectorXcd coefficients(const std::vector<uint32_t>& indices) const;
		Eigen::VectorXcd coefficients() const;
    double trace() const;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) override;
		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		double mzr_prob(uint32_t q, bool outcome) const;

		virtual std::vector<double> probabilities() const override {
			Statevector statevector(*this);
			return statevector.probabilities();
		}
		virtual bool mzr(uint32_t q) override;
    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits);
    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta);
};

class MatrixProductOperatorImpl;

class MatrixProductOperator : public QuantumState {
  private:
    std::unique_ptr<MatrixProductOperatorImpl> impl;

  public:
    MatrixProductOperator()=default;
    ~MatrixProductOperator();

    MatrixProductOperator(const MatrixProductState& mps, const std::vector<uint32_t>& traced_qubits);
    MatrixProductOperator(const MatrixProductOperator& mpo, const std::vector<uint32_t>& traced_qubits);
    MatrixProductOperator(const MatrixProductOperator& other);

    void print_mps() const;
		Eigen::MatrixXcd coefficients() const;
    double trace() const;

    virtual std::complex<double> expectation(const PauliString& p) const override;

    virtual std::vector<double> probabilities() const override {
      return DensityMatrix(coefficients()).probabilities();
    }

    virtual std::string to_string() const override {
      return DensityMatrix(coefficients()).to_string();
    }

    MatrixProductOperator partial_trace(const std::vector<uint32_t>& qubits) const;

    virtual magic_t magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;
    virtual magic_t magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) override;
    virtual magic_t magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override {
      throw std::runtime_error("entropy not implemented for MatrixProductOperator.");
    }

    virtual bool mzr(uint32_t q) override {
      throw std::runtime_error("mzr not implemented for MatrixProductOperator.");
    }

    virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) override {
      throw std::runtime_error("evolve not implemented for MatrixProductOperator.");
    }
};

static std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> retrieve_traced_qubits(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_qubits) {
  std::vector<bool> mask(num_qubits, false);

  for (const auto q : qubitsA) {
    mask[q] = true;
  }

  for (const auto q : qubitsB) {
    mask[q] = true;
  }

  // Trace out qubits not in A or B
  std::vector<uint32_t> _qubits;
  for (size_t i = 0; i < num_qubits; i++) {
    if (!mask[i]) {
      _qubits.push_back(i);
    }
  }

  std::vector<size_t> offset(num_qubits);
  size_t k = 0;
  for (size_t i = 0; i < num_qubits; i++) {
    if (!mask[i]) {
      k++;
    }
    
    offset[i] = k;
  }

  std::vector<uint32_t> _qubitsA(qubitsA.begin(), qubitsA.end());
  for (size_t i = 0; i < qubitsA.size(); i++) {
    _qubitsA[i] -= offset[_qubitsA[i]];
  }

  std::vector<uint32_t> _qubitsB(qubitsB.begin(), qubitsB.end());
  for (size_t i = 0; i < qubitsB.size(); i++) {
    _qubitsB[i] -= offset[_qubitsB[i]];
  }

  return {_qubits, _qubitsA, _qubitsB};
}

template <class StateType>
magic_t magic_mutual_information_exhaustive_impl(StateType& state, const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) {
  auto [_qubits, _qubitsA, _qubitsB] = retrieve_traced_qubits(qubitsA, qubitsB, state.num_qubits);

  std::vector<bool> maskA(state.num_qubits, true);
  for (auto q : qubitsA) {
    maskA[q] = false;
  }

  std::vector<bool> maskB(state.num_qubits, true);
  for (auto q : qubitsB) {
    maskB[q] = false;
  }

  auto stateAB = state.partial_trace(_qubits);
  auto stateA = stateAB.partial_trace(_qubitsB);
  auto stateB = stateAB.partial_trace(_qubitsA);

  auto samplesA = stateB.sample_paulis_exhaustive();
  auto samplesB = stateB.sample_paulis_exhaustive();
  auto samplesAB = stateAB.sample_paulis_exhaustive();

  auto power = [](double s, const PauliAmplitude& p, double pow) {
    return s + std::pow(p.second, pow);
  };

  auto power2 = std::bind(power, std::placeholders::_1, std::placeholders::_2, 2.0);
  auto power4 = std::bind(power, std::placeholders::_1, std::placeholders::_2, 4.0);

  auto power_vec = [&power](const std::vector<PauliAmplitude>& samples, double pow) {
    auto powfunc = std::bind(power, std::placeholders::_1, std::placeholders::_2, pow);
    return std::accumulate(samples.begin(), samples.end(), 0.0, powfunc);
  };

  double sumA_2 = power_vec(samplesA, 2.0);
  double sumA_4 = power_vec(samplesA, 4.0);
  double sumB_2 = power_vec(samplesB, 2.0);
  double sumB_4 = power_vec(samplesB, 4.0);
  double sumAB_2 = power_vec(samplesAB, 2.0);
  double sumAB_4 = power_vec(samplesAB, 4.0);

  double I = -std::log(sumA_2*sumB_2/sumAB_2);
  double W = -std::log(sumA_4*sumB_4/sumAB_4);

  return I - W;
}

static void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng, size_t j) {
  size_t g = rng() % 4;

  bool b1 = g % 2;
  bool b2 = g < 2;
  
  if (g == 0) {
    p.set_x(j, 0);
    p.set_z(j, 0);
  } else if (g == 1) {
    p.set_x(j, 1);
    p.set_z(j, 0);
  } else if (g == 2) {
    p.set_x(j, 0);
    p.set_z(j, 1);
  } else {
    p.set_x(j, 1);
    p.set_z(j, 1);
  }
}

static void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % p.num_qubits;
  single_qubit_random_mutation(p, rng, j);
}

static void random_mutation(PauliString& p, std::minstd_rand& rng) {
  bool r = rng() % 2;
  if ((r) || (p.num_qubits == 1)) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    single_qubit_random_mutation(p, rng, j);
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2 = rng() % p.num_qubits;
    while (j2 == j1) {
      j2 = rng() % p.num_qubits;
    }

    single_qubit_random_mutation(p, rng, j1);
    single_qubit_random_mutation(p, rng, j2);
  }
}

static void xxz_random_mutation(PauliString& p, std::minstd_rand& rng) {
  PauliString pnew(p);
  if ((rng() % 2) || (p.num_qubits == 1)) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    PauliString Zj = PauliString(p.num_qubits);
    Zj.set_z(j, 1); 

    pnew *= Zj;
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2 = rng() % p.num_qubits;
    while (j2 == j1) {
      j2 = rng() % p.num_qubits;
    }

    PauliString Xij = PauliString(p.num_qubits);
    Xij.set_x(j1, 1); 
    Xij.set_x(j2, 1); 
    pnew *= Xij;
  }

  p = pnew;
}

static void global_random_mutation(PauliString& p, std::minstd_rand& rng) {
  p = PauliString::rand(p.num_qubits, rng);
}

static void random_bit_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % (2*p.num_qubits);
  p.set(j, !p.get(j));
}

template <class StateType>
magic_t magic_mutual_information_montecarlo_impl(StateType& state, const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) {
  PauliMutationFunc mutation = xxz_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto [_qubits, _qubitsA, _qubitsB] = retrieve_traced_qubits(qubitsA, qubitsB, state.num_qubits);

  auto stateAB = state.partial_trace(_qubits);
  auto stateA = stateAB.partial_trace(_qubitsB);
  auto stateB = stateAB.partial_trace(_qubitsA);

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples1 = stateAB.sample_paulis_montecarlo(num_samples, equilibration_timesteps, p1, mutation);
  auto samples2 = stateAB.sample_paulis_montecarlo(num_samples, equilibration_timesteps, p2, mutation);

  double I = 0.0;
  for (const auto &[P, t] : samples1) {
    PauliString PA = P.substring(_qubitsA, true);
    PauliString PB = P.substring(_qubitsB, true);

    double tA = std::abs(stateA.expectation(PA));
    double tB = std::abs(stateB.expectation(PB));

    I += std::pow(tA*tB/t, 2.0);
  }

  I = -std::log(I/samples1.size());

  double W = 0.0;
  for (const auto &[P, t] : samples2) {
    PauliString PA = P.substring(_qubitsA, true);
    PauliString PB = P.substring(_qubitsB, true);

    double tA = std::abs(stateA.expectation(PA));
    double tB = std::abs(stateB.expectation(PB));

    W += std::pow(tA*tB/t, 4.0);
  }
  
  W = -std::log(W/samples2.size());

  return I - W;
}

template <class StateType>
magic_t magic_mutual_information_exact_impl(StateType& state, const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  auto [_qubits, _qubitsA, _qubitsB] = retrieve_traced_qubits(qubitsA, qubitsB, state.num_qubits);

  auto stateAB = state.partial_trace(_qubits);
  auto stateA = stateAB.partial_trace(_qubitsB);
  auto stateB = stateAB.partial_trace(_qubitsA);

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples1 = stateAB.sample_paulis_exact(num_samples, p1);
  auto samples2 = stateAB.sample_paulis_exact(num_samples, p2);

  double I = 0.0;
  for (const auto &[P, t] : samples1) {
    PauliString PA = P.substring(_qubitsA, true);
    PauliString PB = P.substring(_qubitsB, true);

    double tA = std::abs(stateA.expectation(PA));
    double tB = std::abs(stateB.expectation(PB));

    I += std::pow(tA*tB/t, 2.0);
  }

  I = -std::log(I/samples1.size());

  double W = 0.0;
  for (const auto &[P, t] : samples2) {
    PauliString PA = P.substring(_qubitsA, true);
    PauliString PB = P.substring(_qubitsB, true);

    double tA = std::abs(stateA.expectation(PA));
    double tB = std::abs(stateB.expectation(PB));

    W += std::pow(tA*tB/t, 4.0);
  }
  
  W = -std::log(W/samples2.size());

  return I - W;
}
