#include "QuantumStates.h"
#include <unsupported/Eigen/KroneckerProduct>

DensityMatrix::DensityMatrix(uint32_t num_qubits) : QuantumState(num_qubits) {
	data = Eigen::MatrixXcd::Zero(basis, basis);
	data(0, 0) = 1;
}

DensityMatrix::DensityMatrix(const Statevector& state) : QuantumState(state.num_qubits) {	
	data = Eigen::kroneckerProduct(state.data.adjoint(), state.data);
}

DensityMatrix::DensityMatrix(const QuantumCircuit& circuit) : DensityMatrix(circuit.num_qubits) {
	evolve(circuit);
}

DensityMatrix::DensityMatrix(const DensityMatrix& rho) : QuantumState(rho.num_qubits) {
	data = rho.data;
}

DensityMatrix::DensityMatrix(const UnitaryState& U) : DensityMatrix(U.num_qubits) {
  evolve(U.unitary);
}

DensityMatrix::DensityMatrix(const MatrixProductState& mps) : DensityMatrix(Statevector(mps)) {
}

DensityMatrix::DensityMatrix(const Eigen::MatrixXcd& data) : data(data) {
  size_t nrows = data.rows();
  size_t ncols = data.cols();

  if (nrows != ncols) {
    throw std::runtime_error("Provided data is not square.");
  }

  if (!(nrows > 0 && ((nrows & (nrows - 1)) == 0))) {
    throw std::runtime_error("Provided data does not have a dimension which is a power of 2.");
  } 

  num_qubits = std::bit_width(nrows) - 1;
}

std::string DensityMatrix::to_string() const {
	std::stringstream ss;
	ss << data;
	return fmt::format("DensityMatrix({}):\n{}\n", num_qubits, ss.str());
}

DensityMatrix DensityMatrix::partial_trace(const std::vector<uint32_t>& traced_qubits) const {
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

std::vector<double> DensityMatrix::magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  std::vector<uint32_t> qubitsAB;

  qubitsAB.insert(qubitsAB.end(), qubitsA.begin(), qubitsA.end());
  qubitsAB.insert(qubitsAB.end(), qubitsB.begin(), qubitsB.end());
  std::vector<uint32_t> qubitsA_(qubitsA.size());
  std::vector<uint32_t> qubitsB_(qubitsB.size());
  std::iota(qubitsA_.begin(), qubitsA_.end(), 0);
  std::iota(qubitsB_.begin(), qubitsB_.end(), 0);

  DensityMatrix rhoAB = partial_trace(qubitsAB);
  auto samples = rhoAB.stabilizer_renyi_entropy_samples(num_samples);
  std::vector<double> magic_samples;
  for (const auto& [P, p] : samples) {
    PauliString PA = P.substring(qubitsA_);
    PauliString PB = P.substring(qubitsB_);

    double tAB = std::abs(rhoAB.expectation(P));
    double tA = std::abs(rhoAB.expectation(PA));
    double tB = std::abs(rhoAB.expectation(PB));

    magic_samples.push_back(tAB/(tA*tB));
  }

  return magic_samples;
}

double DensityMatrix::entropy(const std::vector<uint32_t> &qubits, uint32_t index) {
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

std::complex<double> DensityMatrix::expectation(const PauliString &p) const {
  Eigen::MatrixXcd P = p.to_matrix();
  return (data*P).trace();
}

void DensityMatrix::evolve(const Eigen::MatrixXcd& gate) {
	data = gate * data * gate.adjoint();
}

void DensityMatrix::evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) {
	evolve(full_circuit_unitary(gate, qbits, num_qubits));
}

void DensityMatrix::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	QuantumState::evolve(gate, qubit);
}

void DensityMatrix::evolve_diagonal(const Eigen::VectorXcd& gate, const std::vector<uint32_t>& qbits) {
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

void DensityMatrix::evolve_diagonal(const Eigen::VectorXcd& gate) {
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

bool DensityMatrix::measure(uint32_t q) {
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

std::vector<bool> DensityMatrix::measure_all() {
	for (uint32_t i = 0; i < basis; i++) {
		for (uint32_t j = 0; j < basis; j++) {
			if (i != j) {
				data(i, j) = 0;
			}
		}
	}

	return std::vector<bool>(num_qubits, 0);
}


Eigen::VectorXd DensityMatrix::diagonal() const {
	return data.diagonal().cwiseAbs();
}

std::map<uint32_t, double> DensityMatrix::probabilities_map() const {
	std::map<uint32_t, double> outcomes;

	for (uint32_t i = 0; i < basis; i++) {
		outcomes[i] = data(i, i).real();
	}

	return outcomes;
}

std::vector<double> DensityMatrix::probabilities() const {
	std::vector<double> probs(basis);
	for (uint32_t i = 0; i < basis; i++) {
		probs[i] = data(i, i).real();
	}

	return probs;
}
