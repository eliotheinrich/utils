#include "QuantumCircuit.h"
#include "EntropyState.hpp"

#include <map>
#include <bitset>
#include <iostream>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "utils.hpp"

#define QS_ATOL 1e-8

using PauliAmplitude = std::pair<PauliString, double>;
using PauliMutationFunc = std::function<void(PauliString&, std::minstd_rand&)>;
using ProbabilityFunc = std::function<double(double)>;
using WeakMeasurementData = std::tuple<PauliString, std::vector<uint32_t>, double>;
using MeasurementData = std::tuple<PauliString, std::vector<uint32_t>>;
using MeasurementOutcome = std::tuple<Eigen::MatrixXcd, double, bool>;

class QuantumState : public EntropyState, public std::enable_shared_from_this<QuantumState> {
	protected:
    std::minstd_rand rng;

	public:
    uint32_t rand() { 
      return this->rng(); 
    }

    double randf() { 
      return double(rand())/double(RAND_MAX); 
    }
		uint32_t num_qubits;
		uint32_t basis;

		QuantumState()=default;
    ~QuantumState()=default;

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

    virtual double magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
      throw std::runtime_error("Virtual magic_mutual_information called on a state which has not implemented it.");
    }
    virtual double magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual double magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB);
    virtual double magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples);

    virtual std::vector<double> bipartite_magic_mutual_information(size_t num_samples);
    virtual std::vector<double> bipartite_magic_mutual_information_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<double> bipartite_magic_mutual_information_exhaustive();
    virtual std::vector<double> bipartite_magic_mutual_information_exact(size_t num_samples);

    static double stabilizer_renyi_entropy(size_t index, const std::vector<PauliAmplitude>& samples);

    std::vector<PauliAmplitude> sample_paulis_montecarlo(size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    std::vector<PauliAmplitude> sample_paulis_exhaustive();
    std::vector<PauliAmplitude> sample_paulis_exact(size_t num_samples, ProbabilityFunc prob);

    virtual std::vector<PauliAmplitude> sample_paulis(size_t num_samples) {
      ProbabilityFunc prob = [](double t) -> double { return std::pow(t, 2.0); };
      return sample_paulis_montecarlo(num_samples, 5*num_qubits, prob);
    }

    virtual std::shared_ptr<QuantumState> partial_trace(const std::vector<uint32_t>& qubits) const=0;

    virtual double expectation(const PauliString& p) const=0;

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
    DEFINE_ONE_QUBIT_GATE(t, T);       \
    DEFINE_ONE_QUBIT_GATE(td, Td);     \

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

    DensityMatrix(const MatrixProductState& mps);

    DensityMatrix(const MatrixProductOperator& mpo);

		DensityMatrix(const Eigen::MatrixXcd& data);

		virtual std::string to_string() const override;

		DensityMatrix partial_trace_density_matrix(const std::vector<uint32_t>& traced_qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const std::vector<uint32_t>& qubits) const override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override;

    virtual double expectation(const PauliString& p) const override;
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

    virtual std::shared_ptr<QuantumState> partial_trace(const std::vector<uint32_t>& qubits) const override;

    virtual double expectation(const PauliString& p) const override;
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

    void internal_measure(const MeasurementOutcome& outcome, const std::vector<uint32_t>& qubits, bool renormalize);

    MeasurementOutcome measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits);
    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits);
    std::vector<bool> measure(const std::vector<MeasurementData>& measurements);

    MeasurementOutcome weak_measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits, double beta);
    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta);
    std::vector<bool> weak_measure(const std::vector<WeakMeasurementData>& measurements);

		double norm() const;

		void normalize();

		void fix_gauge();

		double probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const;

		virtual std::vector<double> probabilities() const override;

		std::map<uint32_t, double> probabilities_map() const;

		std::complex<double> inner(const Statevector& other) const;

		Eigen::VectorXd svd(const std::vector<uint32_t>& qubits) const;
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
    MatrixProductState(const Statevector& other, uint32_t bond_dimension, double sv_threshold=1e-8);
    MatrixProductState& operator=(const MatrixProductState& other);

    static MatrixProductState ising_ground_state(size_t num_qubits, double h, size_t bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);

		virtual std::string to_string() const override;

		virtual double entropy(const std::vector<uint32_t>& qubits, uint32_t index) override;

    virtual std::vector<PauliAmplitude> sample_paulis(size_t num_samples) override;

    virtual double magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) override;

    //virtual std::vector<double> bipartite_magic_mutual_information(size_t num_samples) override;
    virtual std::vector<double> bipartite_magic_mutual_information_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;

    MatrixProductOperator partial_trace_mpo(const std::vector<uint32_t>& qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const std::vector<uint32_t>& qubits) const override;

    virtual double expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const;
    std::vector<double> pauli_expectation_left_sweep(const PauliString& P, uint32_t q1, uint32_t q2) const;
    std::vector<double> pauli_expectation_right_sweep(const PauliString& P, uint32_t q1, uint32_t q2) const;

		std::complex<double> coefficients(uint32_t z) const;
		Eigen::VectorXcd coefficients(const std::vector<uint32_t>& indices) const;
		Eigen::VectorXcd coefficients() const;
    double trace() const;
    size_t bond_dimension(size_t i) const;

    void reverse();

    std::complex<double> inner(const MatrixProductState& other) const;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) override;
		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void id(uint32_t q1, uint32_t q2);

		virtual std::vector<double> probabilities() const override {
			Statevector statevector(*this);
			return statevector.probabilities();
		}

		virtual bool mzr(uint32_t q) override;

    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits);
    std::vector<bool> measure(const std::vector<MeasurementData>& measurements);

    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta);
    std::vector<bool> weak_measure(const std::vector<WeakMeasurementData>& measurements);

		void print_mps(bool print_data) const;

    void id_debug(uint32_t i, uint32_t j);
    std::vector<size_t> orthogonal_sites() const;
    void show_problem_sites() const;
    bool debug_tests();
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
    MatrixProductOperator& operator=(const MatrixProductOperator& other);

    void print_mps() const;
		Eigen::MatrixXcd coefficients() const;
    double trace() const;

    virtual double expectation(const PauliString& p) const override;

    virtual std::vector<double> probabilities() const override {
      return DensityMatrix(coefficients()).probabilities();
    }

    virtual std::string to_string() const override {
      return DensityMatrix(coefficients()).to_string();
    }

    virtual std::shared_ptr<QuantumState> partial_trace(const std::vector<uint32_t>& qubits) const override;
    MatrixProductOperator partial_trace_mpo(const std::vector<uint32_t>& qubits) const;
    virtual std::vector<PauliAmplitude> sample_paulis(size_t num_samples) override;

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

double calculate_magic(
  const std::vector<double>& tA2, const std::vector<double>& tB2, const std::vector<double> tAB2,
  const std::vector<double>& tA4, const std::vector<double>& tB4, const std::vector<double> tAB4
);

void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng);

