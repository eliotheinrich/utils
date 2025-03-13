#include "QuantumStates.h"
#include <unsupported/Eigen/KroneckerProduct>

#include <glaze/glaze.hpp>

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

DensityMatrix::DensityMatrix(const MatrixProductState& mps) : QuantumState(mps.num_qubits) {
  if (mps.is_pure_state()) {
    Statevector state(mps.coefficients_pure());
	  data = Eigen::kroneckerProduct(state.data.adjoint(), state.data);
  } else {
    data = mps.coefficients_mixed();
  }
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

DensityMatrix DensityMatrix::partial_trace_density_matrix(const Qubits& qubits) const {
  auto interval = support_range(qubits);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > num_qubits) {
      throw std::runtime_error(fmt::format("qubits = {} passed to DensityMatrix.partial_trace with {} qubits", qubits, num_qubits));
    }
  }
  
	Qubits remaining_qubits;
	for (uint32_t q = 0; q < num_qubits; q++) {
		if (!std::count(qubits.begin(), qubits.end(), q)) {
			remaining_qubits.push_back(q);
		}
	}

	uint32_t num_traced_qubits = qubits.size();
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
					uint32_t q = qubits[k];
					i = quantumstate_utils::set_bit(i, q, n, k);
					j = quantumstate_utils::set_bit(j, q, n, k);
				}

				reduced_rho.data(r, c) += data(i, j);
			}
		}
	}

	return reduced_rho;
}

// TODO make sure that this is a move
std::shared_ptr<QuantumState> DensityMatrix::partial_trace(const Qubits& qubits) const {
  return std::make_shared<DensityMatrix>(std::move(partial_trace_density_matrix(qubits)));
}

double DensityMatrix::entropy(const std::vector<uint32_t>& qubits, uint32_t index) {
	// If number of qubits is larger than half the system, take advantage of the fact that 
	// S_A = S_\bar{A} to compute entropy for the smaller of A and \bar{A}
	if (qubits.size() > num_qubits/2) {
		Qubits qubits_complement;
    std::vector<bool> mask(num_qubits, true);
    for (const auto q : qubits) {
      mask[q] = false;
    }

		for (uint32_t q = 0; q < num_qubits; q++) {
      if (mask[q]) {
        qubits_complement.push_back(q);
      }
		}

		return entropy(qubits_complement, index);
	}

	Qubits traced_qubits;
	for (uint32_t q = 0; q < num_qubits; q++) {
		if (!std::count(qubits.begin(), qubits.end(), q)) {
			traced_qubits.push_back(q);
		}
	}


	DensityMatrix rho_a = partial_trace_density_matrix(traced_qubits);

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
				s -= eigenvalue*std::log(eigenvalue);
			}
		}
		return s;
	} else {
		return 1./(1. - index) * std::log(rho_a.data.pow(index).trace().real());
	}
}

std::complex<double> DensityMatrix::expectation(const PauliString &p) const {
  if (p.num_qubits == 0) {
    return 1.0;
  }

  Eigen::MatrixXcd P = p.to_matrix();
  return expectation(P);
}

std::complex<double> DensityMatrix::expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const {
  Eigen::MatrixXcd M = full_circuit_unitary(m, qubits, num_qubits);
  return expectation(M);
}

std::complex<double> DensityMatrix::expectation(const Eigen::MatrixXcd& m) const {
  size_t r = m.rows();
  size_t c = m.cols();

  if ((r != c) || (1u << num_qubits != r)) {
    throw std::runtime_error(fmt::format("Expectation of provided {}x{} matrix cannot be calculated for DensityMatrix of {} qubits.", r, c, num_qubits));
  }

  return (data * m).trace();
}

void DensityMatrix::evolve(const Eigen::MatrixXcd& gate) {
	data = gate * data * gate.adjoint();
}

void DensityMatrix::evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  assert_gate_shape(gate, qubits);
	evolve(full_circuit_unitary(gate, qubits, num_qubits));
}

void DensityMatrix::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	QuantumState::evolve(gate, qubit);
}

void DensityMatrix::evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) {
	uint32_t s = 1u << num_qubits;
	uint32_t h = 1u << qubits.size();

	if (gate.size() != h) {
		throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
	}

	for (uint32_t a1 = 0; a1 < s; a1++) {
		for (uint32_t a2 = 0; a2 < s; a2++) {
			uint32_t b1 = quantumstate_utils::reduce_bits(a1, qubits);
			uint32_t b2 = quantumstate_utils::reduce_bits(a2, qubits);

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

bool DensityMatrix::mzr(uint32_t q) {
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

double DensityMatrix::mzr_prob(uint32_t q, bool outcome) const {
  double prob_one = 0.0;
  for (uint32_t i = 0; i < basis; i++) {
    if ((i >> q) & 1u) {
      prob_one += std::abs(data(i, i));
    }
  }

  if (outcome) {
    return prob_one;
  } else {
    return 1.0 - prob_one;
  }
}

bool DensityMatrix::forced_mzr(uint32_t q, bool outcome) {
  double prob_zero = mzr_prob(q, 0);
  check_forced_measure(outcome, prob_zero);

	for (uint32_t i = 0; i < basis; i++) {
		for (uint32_t j = 0; j < basis; j++) {
			uint32_t q1 = (i >> q) & 1u;
			uint32_t q2 = (j >> q) & 1u;

			if (q1 != outcome || q2 != outcome) {
				data(i, j) = 0;
			}
		}
	}

  data /= trace();

  return outcome;
}


bool DensityMatrix::measure(const Measurement& m) {
  Qubits qubits = m.qubits;
  if (m.is_basis()) { // Special, more efficient behavior for computational basis measurements
    if (m.is_forced()) {
      return forced_mzr(qubits[0], m.get_outcome());
    } else {
      return mzr(qubits[0]);
    }
  }

  PauliString p = m.get_pauli();

  PauliString p_ = p.superstring(qubits, num_qubits);
  Eigen::MatrixXcd matrix = p_.to_matrix();
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(basis, basis);

  if (m.is_forced()) {
    bool outcome = m.get_outcome();

    Eigen::MatrixXcd P;
    double prob_zero;
    if (outcome) {
      P = (id - matrix)/2.0;
      prob_zero = 1.0 - std::abs(expectation(P));
    } else {
      P = (id + matrix)/2.0;
      prob_zero = std::abs(expectation(P));
    }

    bool invalid = check_forced_measure(outcome, prob_zero);
    if (invalid) {
      P = id - P;
    }

    data = P*data*P.adjoint();
    data /= trace();
    return outcome;
  } else { // Unforced; result is mixed state
    Eigen::MatrixXcd P0 = (id - matrix)/2.0;
    Eigen::MatrixXcd P1 = (id + matrix)/2.0;

    double prob = std::abs(expectation(P0));

    data = P0*data*P0.adjoint()/std::sqrt(prob) + P1*data*P1.adjoint()/std::sqrt(1.0 - prob);
    return 0; // No definite outcome; default to 0
  }
}

bool DensityMatrix::weak_measure(const WeakMeasurement& m) {
  Qubits qubits = m.qubits;
  PauliString pauli = m.get_pauli();
  double beta = m.beta;

  PauliString pauli_ = pauli.superstring(qubits, num_qubits);
  Eigen::MatrixXcd matrix = m.beta * pauli_.to_matrix();

  if (m.is_forced()) {
    bool outcome = m.get_outcome();
    Eigen::MatrixXcd P;
    if (outcome) {
      P = (-matrix).exp();
    } else {
      P = matrix.exp();
    }

    // TODO check this
    data = P*data*P.adjoint();
    data /= trace();
    return outcome;
  } else {
    Eigen::MatrixXcd P0 = matrix.exp();
    Eigen::MatrixXcd P1 = (-matrix).exp();

    double prob0 = std::abs(expectation(P0));
    double prob1 = std::abs(expectation(P1));

    // TODO check this
    data = P0*data*P0.adjoint()/std::sqrt(prob0) + P1*data*P1.adjoint()/std::sqrt(prob1);
    return 0;
  }
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

namespace glz::detail {
   template <>
   struct from<BEVE, Eigen::MatrixXcd> {
      template <auto Opts>
      static void op(Eigen::MatrixXcd& value, auto&&... args) {
        std::string str;
        read<BEVE>::op<Opts>(str, args...);
        std::vector<char> buffer(str.begin(), str.end());

        const size_t header_size = 2 * sizeof(size_t);
        size_t rows, cols;
        memcpy(&rows, buffer.data(), sizeof(size_t));
        memcpy(&cols, buffer.data() + sizeof(size_t), sizeof(size_t));
        
        Eigen::MatrixXcd matrix(rows, cols);
        memcpy(matrix.data(), buffer.data() + header_size, rows * cols * sizeof(std::complex<double>));

        value = matrix; 
      }
   };

   template <>
   struct to<BEVE, Eigen::MatrixXcd> {
      template <auto Opts>
      static void op(const Eigen::MatrixXcd& value, auto&&... args) noexcept {
        const size_t header_size = 2 * sizeof(size_t);
        const size_t rows = value.rows();
        const size_t cols = value.cols();
    
        const size_t data_size = rows * cols * sizeof(std::complex<double>);
        std::vector<char> buffer(header_size + data_size);
    
        memcpy(buffer.data(), &rows, sizeof(size_t));
        memcpy(buffer.data() + sizeof(size_t), &cols, sizeof(size_t));
        memcpy(buffer.data() + header_size, value.data(), data_size);

        std::string data(buffer.begin(), buffer.end());
        write<BEVE>::op<Opts>(data, args...);
      }
   };
}

struct DensityMatrix::glaze {
  using T = DensityMatrix;
  static constexpr auto value = glz::object(
    &T::data,
    &T::use_parent,
    &T::num_qubits,
    &T::basis
  );
};

std::vector<char> DensityMatrix::serialize() const {
  std::vector<char> bytes;
  auto write_error = glz::write_beve(*this, bytes);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing DensityMatrix to binary: \n{}", glz::format_error(write_error, bytes)));
  }
  return bytes;
}

void DensityMatrix::deserialize(const std::vector<char>& bytes) {
  auto parse_error = glz::read_beve(*this, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error reading DensityMatrix from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
