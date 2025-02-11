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

class QuantumState;

using PauliAmplitudes = std::pair<PauliString, std::vector<double>>;

using PartialState = std::pair<std::shared_ptr<QuantumState>, QubitSupport>;

using PauliMutationFunc = std::function<void(PauliString&, std::minstd_rand&)>;
using ProbabilityFunc = std::function<double(double)>;

using MutualMagicAmplitudes = std::vector<std::vector<double>>; // tA, tB, tAB
using MutualMagicData = std::pair<MutualMagicAmplitudes, MutualMagicAmplitudes>; // t2, t4

using WeakMeasurementData = std::tuple<PauliString, Qubits, double>;
using MeasurementData = std::tuple<PauliString, Qubits>;
using MeasurementOutcome = std::tuple<Eigen::MatrixXcd, double, bool>;

class QuantumState : public EntropyState, public std::enable_shared_from_this<QuantumState> {
	protected:
    static thread_local std::minstd_rand rng;

    static uint32_t rand() { 
      return QuantumState::rng(); 
    }

    static double randf() { 
      return static_cast<double>(QuantumState::rand())/static_cast<double>(RAND_MAX); 
    }

	public:

		static void seed(int s) {
      QuantumState::rng.seed(s);
		}

    static unsigned int random_seed() {
      thread_local std::random_device gen;
      return gen();
    }


		uint32_t num_qubits;
		uint32_t basis;

		QuantumState()=default;
    ~QuantumState()=default;

		QuantumState(uint32_t num_qubits) : EntropyState(num_qubits), num_qubits(num_qubits), basis(1u << num_qubits) {}

    std::vector<PartialState> get_partial_states(const std::vector<QubitSupport>& qubits) const;

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const=0;
    virtual std::shared_ptr<QuantumState> partial_trace(const QubitSupport& support) const {
      return partial_trace(to_qubits(support));
    }

    virtual std::complex<double> expectation(const PauliString& p) const=0;

		virtual std::string to_string() const=0;

		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits)=0;

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

    void random_clifford(const Qubits& qubits) {
      random_clifford_impl(qubits, QuantumState::rng, *this);
    }

		virtual void evolve(const Eigen::MatrixXcd& gate) {
			Qubits qubits(num_qubits);
			std::iota(qubits.begin(), qubits.end(), 0);
			evolve(gate, qubits);
		}

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t q) {
			Qubits qubit{q};
			evolve(gate, qubit); 
		}

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) { 
			evolve(Eigen::MatrixXcd(gate.asDiagonal()), qubits); 
		}

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) { 
			evolve(Eigen::MatrixXcd(gate.asDiagonal())); 
		}

		virtual void evolve(const Measurement& measurement) {
			for (auto q : measurement.qubits) {
				mzr(q);
			}
		}

		virtual void evolve(const Instruction& inst) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
					evolve(gate->define(), gate->qubits); 
				},
				[this](Measurement m) { 
					for (auto const &q : m.qubits) {
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

    virtual void evolve(const QuantumCircuit& qc, const Qubits& qubits) {
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
    virtual double purity() const=0;

    //                                                                                                          //
    // ---------------------------- STABILIZER RENYI ENTROPY FUNCTIONS ---------------------------------------- //
    //                                                                                                          //
    virtual std::vector<PauliAmplitudes> sample_paulis_montecarlo(const std::vector<QubitSupport>& qubits, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<PauliAmplitudes> sample_paulis_exhaustive(const std::vector<QubitSupport>& qubits);
    virtual std::vector<PauliAmplitudes> sample_paulis_exact(const std::vector<QubitSupport>& qubits, size_t num_samples, ProbabilityFunc prob);

    virtual std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& qubits, size_t num_samples) {
      throw std::runtime_error("Attempted to call virtual sample_paulis on state which does not provide an implementation.");
    }

    double stabilizer_renyi_entropy(size_t index, const std::vector<double>& samples) const;

    static double calculate_magic_mutual_information_from_samples(const MutualMagicAmplitudes& samples2, const MutualMagicAmplitudes& samples4);
    static double calculate_magic_mutual_information_from_samples(const MutualMagicData& data) { return calculate_magic_mutual_information_from_samples(data.first, data.second); }
    static double calculate_magic_mutual_information_from_samples2(const MutualMagicAmplitudes& samples2);

    virtual double magic_mutual_information(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples) {
      throw std::runtime_error("Virtual magic_mutual_information called on a state which does not provide an implementation.");
    }
    virtual MutualMagicData magic_mutual_information_samples_montecarlo(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual double magic_mutual_information_montecarlo(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual MutualMagicData magic_mutual_information_samples_exact(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples);
    virtual double magic_mutual_information_exact(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples);
    virtual double magic_mutual_information_exhaustive(const Qubits& qubitsA, const Qubits& qubitsB);

    virtual std::vector<double> bipartite_magic_mutual_information(size_t num_samples);
    virtual std::vector<MutualMagicData> bipartite_magic_mutual_information_samples_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<double> bipartite_magic_mutual_information_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<MutualMagicData> bipartite_magic_mutual_information_samples_exact(size_t num_samples);
    virtual std::vector<double> bipartite_magic_mutual_information_exact(size_t num_samples);
    virtual std::vector<double> bipartite_magic_mutual_information_exhaustive();

    //                                                                                                          //
    // -------------------------------------------------------------------------------------------------------- //
    //                                                                                                          //
};

class DensityMatrix;
class Statevector;
class MatrixProductState;
class MatrixProductMixedState;

class DensityMatrix : public QuantumState {
	public:
		Eigen::MatrixXcd data;

    DensityMatrix()=default;

		DensityMatrix(uint32_t num_qubits);

		DensityMatrix(const Statevector& state);

		DensityMatrix(const QuantumCircuit& circuit);

		DensityMatrix(const DensityMatrix& rho);

    DensityMatrix(const MatrixProductState& mps);

    DensityMatrix(const MatrixProductMixedState& mpo);

		DensityMatrix(const Eigen::MatrixXcd& data);

		virtual std::string to_string() const override;

		DensityMatrix partial_trace_density_matrix(const Qubits& traced_qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

		virtual double entropy(const std::vector<uint32_t> &qubits, uint32_t index) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;

		virtual void evolve(const Eigen::MatrixXcd& gate) override;

		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

		virtual bool mzr(uint32_t q) override;

		virtual std::vector<bool> mzr_all() override;

    bool measure(const PauliString& p, const Qubits& qubits);

		Eigen::VectorXd diagonal() const;

		virtual std::vector<double> probabilities() const override;
    virtual double purity() const override {
      return (data*data).trace().real();

    }
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

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;

		virtual void evolve(const Eigen::MatrixXcd &gate, const Qubits& qubits) override;

		virtual void evolve(const Eigen::MatrixXcd &gate) override;
		
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate, const Qubits& qubits) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd &gate) override;

		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void evolve(const QuantumCircuit& circuit, const std::vector<bool>& outcomes);

		double mzr_prob(uint32_t q, bool outcome) const;

		virtual bool mzr(uint32_t q) override;

    bool mzr(uint32_t q, bool outcome);

    void internal_measure(const MeasurementOutcome& outcome, const Qubits& qubits, bool renormalize);

    MeasurementOutcome measurement_outcome(const PauliString& p, const Qubits& qubits);
    bool measure(const PauliString& p, const Qubits& qubits);
    std::vector<bool> measure(const std::vector<MeasurementData>& measurements);

    MeasurementOutcome weak_measurement_outcome(const PauliString& p, const Qubits& qubits, double beta);
    bool weak_measure(const PauliString& p, const Qubits& qubits, double beta);
    std::vector<bool> weak_measure(const std::vector<WeakMeasurementData>& measurements);

		double norm() const;
		void normalize();
		void fix_gauge();

		double probabilities(uint32_t z, const Qubits& qubits) const;
		std::map<uint32_t, double> probabilities_map() const;
		virtual std::vector<double> probabilities() const override;
    virtual double purity() const override { 
      return 1.0; 
    }

		std::complex<double> inner(const Statevector& other) const;

		Eigen::VectorXd svd(const Qubits& qubits) const;
};

class PauliExpectationTree;
class PauliExpectationTreeImpl;

class PauliExpectationTree {
  private:
    std::unique_ptr<PauliExpectationTreeImpl> impl;

  public:
    size_t num_qubits;

    PauliExpectationTree(const MatrixProductState& state, const PauliString& p);
    ~PauliExpectationTree();
    
    std::complex<double> expectation() const;

    std::complex<double> partial_expectation(uint32_t q1, uint32_t q2) const;

    void propogate_pauli(Pauli p, uint32_t q);

    void modify(const PauliString& P);

    std::string to_string() const;

    PauliString to_pauli_string() const;
};

class MatrixProductStateImpl;

class MatrixProductState : public QuantumState {
  friend class PauliExpectationTree;

	private:
    std::unique_ptr<MatrixProductStateImpl> impl;

	public:
    MatrixProductState()=default;
    ~MatrixProductState();

		MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold=1e-8);
    MatrixProductState(const MatrixProductState& other);
    MatrixProductState(const Statevector& other, uint32_t bond_dimension, double sv_threshold=1e-8);
    MatrixProductState(const std::unique_ptr<MatrixProductStateImpl>& impl);
    MatrixProductState& operator=(const MatrixProductState& other);

    static MatrixProductState ising_ground_state(size_t num_qubits, double h, size_t bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);

		virtual std::string to_string() const override;

		virtual double entropy(const std::vector<uint32_t>& qubits, uint32_t index) override;

    virtual std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& qubits, size_t num_samples) override;
    virtual std::vector<PauliAmplitudes> sample_paulis_montecarlo(const std::vector<QubitSupport>& qubits, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;

    virtual double magic_mutual_information(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples) override;
    virtual double magic_mutual_information_montecarlo(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;

    virtual std::vector<double> bipartite_magic_mutual_information(size_t num_samples) override;
    virtual std::vector<double> bipartite_magic_mutual_information_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt) override;

    MatrixProductState partial_trace_mps(const Qubits& qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;

    bool is_pure_state() const;
    Eigen::MatrixXcd coefficients_mixed() const;
    Eigen::VectorXcd coefficients_pure() const;
    double trace() const;
    size_t bond_dimension(size_t i) const;

    void reverse();

    std::complex<double> inner(const MatrixProductState& other) const;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;
		virtual void evolve(const QuantumCircuit& circuit) override { 
			QuantumState::evolve(circuit); 
		}

    void id(uint32_t q1, uint32_t q2);

		virtual std::vector<double> probabilities() const override {
			Statevector statevector(*this);
			return statevector.probabilities();
		}

    virtual double purity() const override;

		virtual bool mzr(uint32_t q) override;

    bool measure(const PauliString& p, const Qubits& qubits);
    std::vector<bool> measure(const std::vector<MeasurementData>& measurements);

    bool weak_measure(const PauliString& p, const Qubits& qubits, double beta);
    std::vector<bool> weak_measure(const std::vector<WeakMeasurementData>& measurements);

		void print_mps(bool print_data=false) const;

    void id_debug(uint32_t i, uint32_t j);
    std::vector<size_t> orthogonal_sites() const;
    void show_problem_sites() const;
    bool debug_tests();
};

void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng);

std::vector<QubitSupport> get_bipartite_supports(size_t num_qubits);

std::tuple<Qubits, Qubits, Qubits> get_traced_qubits(
  const Qubits& qubitsA, const Qubits& qubitsB, size_t num_qubits
);

std::vector<std::vector<double>> extract_amplitudes(const std::vector<PauliAmplitudes>& pauli_samples);

inline std::array<std::vector<double>, 3> unfold_mutual_magic_amplitudes(const MutualMagicAmplitudes& samples) {
  return {samples[0], samples[1], samples[2]};
}

