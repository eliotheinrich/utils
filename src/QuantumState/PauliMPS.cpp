#include "QuantumStates.h"

#include <memory>
#include <sstream>
#include <random>

#include <fmt/ranges.h>
#include <itensor/all.h>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

using namespace itensor;

ITensor vector_to_tensor(const Eigen::VectorXcd& v, const std::vector<Index>& idxs) {
  size_t num_qubits = idxs.size();
  if (num_qubits != std::log2(v.size())) {
    throw std::runtime_error(fmt::format("Mismatched number of qubits passed to vector_to_tensor. idxs.size() = {}, log2(v.size()) = {}\n", idxs.size(), std::log2(v.size())));
  }

  for (const auto& idx : idxs) {
    if (dim(idx) != 2) {
      throw std::runtime_error("Every index must have dim = 2.");
    }
  }

  if (num_qubits > 31) {
    throw std::runtime_error("Too many qubits. Must be num_qubits < 32.");
  }

  uint32_t s = 1u << num_qubits;

  ITensor C(idxs);

  for (uint32_t i = 0; i < s; i++) {
    auto bits = quantumstate_utils::to_bits(i, num_qubits);
    std::vector<IndexVal> assignments(num_qubits);
    for (size_t j = 0; j < num_qubits; j++) {
      assignments[j] = (idxs[j] = (static_cast<uint32_t>(bits[j]) + 1));
    }
    C.set(assignments, v[i]);
  }

  return C;
}

ITensor matrix_to_tensor(const Eigen::MatrixXcd& matrix, const std::vector<Index>& idxs1, const std::vector<Index>& idxs2) {
  uint32_t num_idxs = idxs1.size() + idxs2.size();
  uint32_t dim1 = 1u;
  for (size_t i = 0; i < idxs1.size(); i++) {
    dim1 *= dim(idxs1[i]);
  }

  uint32_t dim2 = 1u;
  for (size_t i = 0; i < idxs2.size(); i++) {
    dim2 *= dim(idxs2[i]);
  }

  if ((dim1 != matrix.rows()) || (dim2 != matrix.cols())) {
    throw std::runtime_error("Dimension mismatch in matrix and provided indices!");
  }

  if (num_idxs > 31) {
    throw std::runtime_error("Cannot calculate matrix_to_tensor for more than 31 indices.");
  }

  std::vector<Index> idxs;
  for (size_t i = 0; i < idxs1.size(); i++) {
    idxs.push_back(idxs1[i]);
  }

  for (size_t i = 0; i < idxs2.size(); i++) {
    idxs.push_back(idxs2[i]);
  }

  ITensor tensor(idxs);

  uint32_t s1 = 1u << idxs1.size();
  uint32_t s2 = 1u << idxs2.size();

  for (uint32_t z1 = 0; z1 < s1; z1++) {
    for (uint32_t z2 = 0; z2 < s2; z2++) {
      std::vector<IndexVal> assignments(num_idxs);
      for (uint32_t i = 0; i < idxs1.size(); i++) {
        assignments[i] = (idxs1[i] = ((z1 >> i) & 1u) + 1u);
      }

      for (uint32_t i = 0; i < idxs2.size(); i++) {
        assignments[i + idxs1.size()] = (idxs2[i] = ((z2 >> i) & 1u) + 1u);
      }
      
      tensor.set(assignments, matrix(z1, z2));
    }
  }

  return tensor;
}

double distance_from_identity(const ITensor& A) {
  auto idxs = inds(A);
  Index i1 = idxs[0];
  Index i2 = idxs[1];

  ITensor I(i1, i2);
  for (uint32_t k = 1; k <= dim(i1); k++) {
    I.set(i1=k, i2=k, 1.0);
  }

  return norm(I - A);
}

std::complex<double> tensor_to_scalar(const ITensor& A) {
  std::vector<uint32_t> assignments;
  return eltC(A, assignments);
}

void swap_tags(Index& idx1, Index& idx2) {
  IndexSet idxs = {idx1, idx2};
  auto new_idxs = swapTags(idxs, idx1.tags(), idx2.tags());
  idx1 = new_idxs[0];
  idx2 = new_idxs[1];
}

template <typename T1, typename T2, typename... Args>
void swap_tags(Index& idx1, Index& idx2, T1& first, T2& second, Args&... args) {
  auto tags1 = idx1.tags();
  auto tags2 = idx2.tags();

  first.swapTags(tags1, tags2);
  second.swapTags(tags2, tags1);

  if constexpr (sizeof...(args) == 0) {
    swap_tags(idx1, idx2);
  } else {
    swap_tags(idx1, idx2, args...);
  }
}

template <typename T1, typename T2, typename... Args>
void swap_tags_(const Index& idx1, const Index& idx2, T1& first, T2& second, Args&... args) {
  auto tags1 = idx1.tags();
  auto tags2 = idx2.tags();

  first.swapTags(tags1, tags2);
  second.swapTags(tags2, tags1);

  if constexpr (sizeof...(args) == 0) {
    return;
  } else {
    swap_tags(idx1, idx2, args...);
  }
}

template <typename... Tensors>
void match_indices(const std::string& tags, const ITensor& base, Tensors&... tensors) {
  const auto idx = noPrime(findInds(base, tags)[0]);
  auto do_match = [&idx, &tags](ITensor& tensor) {
    auto like_indices = findInds(tensor, tags);
    for (const auto idx_ : like_indices) {
      tensor.replaceInds({idx_}, {prime(idx, primeLevel(idx_))});
    }
  };

  auto do_match_wrapper = [&do_match](ITensor& tensor) {
    try {
      do_match(tensor);
    } catch (const ITError& error) {
      tensor = toDense(tensor);
      do_match(tensor);
    }
  };

  (do_match_wrapper(tensors), ...);
}

bool is_bipartition(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_qubits) {
  std::vector<bool> mask(num_qubits, false);
  for (const uint32_t q : qubitsA) {
    mask[q] = true;
  }

  for (const uint32_t q : qubitsB) {
    mask[q] = true;
  }

  return std::all_of(mask.begin(), mask.end(), [](bool covered) { return covered; });
}

class PauliMPSImpl {
  friend class PauliMPS;
  public:
		std::vector<ITensor> tensors;
		std::vector<ITensor> singular_values;

    Index left_boundary_index;
    Index right_boundary_index;

		std::vector<Index> external_indices;
		std::vector<Index> internal_indices;

    uint32_t num_qubits;
    uint32_t bond_dimension;
    double sv_threshold;

    PauliMPSImpl()=default;
    ~PauliMPSImpl()=default;

    PauliMPSImpl(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) 
      : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold) {
        if (sv_threshold < 1e-15) {
          throw std::runtime_error("sv_threshold must be finite ( > 0) or else the MPS may be numerically unstable.");
        }

        if ((bond_dimension > 1u << num_qubits) && (num_qubits < 32)) {
          bond_dimension = 1u << num_qubits;
        }

        if (num_qubits < 1) {
          throw std::invalid_argument("Number of qubits must be > 1 for MPS simulator.");
        }

        left_boundary_index = Index(1, "Internal,LEdge");
        right_boundary_index = Index(1, "Internal,REdge");

        for (uint32_t i = 0; i < num_qubits - 1; i++) {
          internal_indices.push_back(Index(1, fmt::format("Internal,n={}", i)));
        }

        for (uint32_t i = 0; i < num_qubits; i++) {
          external_indices.push_back(Index(2, fmt::format("External,i={}", i)));
        }

        ITensor tensor;

        if (num_qubits == 1) {
          tensor = ITensor(left_boundary_index, right_boundary_index, external_idx(0));
          tensor.set(1, 1, 1, 1.0);
          tensors.push_back(tensor);
          return;
        }

        // Setting singular values
        //for (uint32_t q = 0; q < num_qubits - 1; q++) {
        //  tensor = ITensor(internal_idx(q, InternalDir::Left), internal_idx(q, InternalDir::Right));
        //  tensor.set(1, 1, 1.0);
        //  singular_values.push_back(tensor);
        //}

        // Setting bulk tensors
        for (uint32_t q = 1; q < num_qubits - 1; q++) {
          tensor = ITensor(internal_idx(q - 1), internal_idx(q), external_idx(q));
          tensor.set(1, 1, 1, 1.0);
          tensors.push_back(tensor);
        }
      }

    PauliMPSImpl(const PauliMPSImpl& other) : PauliMPSImpl(other.num_qubits, other.bond_dimension, other.sv_threshold) {
      tensors = other.tensors;
      singular_values = other.singular_values;

      left_boundary_index = other.left_boundary_index;
      right_boundary_index = other.right_boundary_index;
      internal_indices = other.internal_indices;
      external_indices = other.external_indices;
    }

    PauliMPSImpl partial_trace(const Qubits& qubits) {
      throw std::runtime_error("Have not yet implemented PauliMPS.partial_trace");
    }

    Index external_idx(size_t i) const {
      if (i >= num_qubits) {
        throw std::runtime_error(fmt::format("Cannot retrieve external index for i = {}.", i));
      }

      return external_indices[i];
    }

    Index internal_idx(size_t i) const {
      if (i >= num_qubits) {
        throw std::runtime_error(fmt::format("Cannot retrieve external index for i = {}.", i));
      }

      return internal_indices[i];
    }

    std::string to_string() {
      Statevector psi(coefficients());
      return psi.to_string();
    }

    void print_mps(bool print_data=false) const {
      auto print_tensor = [&](const ITensor& tensor) {
        if (print_data) {
          PrintData(tensor);
        } else {
          print(tensor);
        }
      };

      print_tensor(tensors[0]);

      for (size_t q = 0; q < num_qubits - 1; q++) {
        //print_tensor(singular_values[q]);
        print_tensor(tensors[q+1]);
      }
    }

    std::vector<std::vector<std::vector<std::complex<double>>>> get_tensor(uint32_t q) const {
      ITensor T = tensors[q];

      Index ext = external_idx(q);

      Index left_idx;
      if (q == 0) {
        left_idx = left_boundary_index;
      } else {
        T *= singular_values[q - 1];
        left_idx = internal_idx(q - 1);
      }
      size_t d1 = dim(left_idx);

      Index right_idx;
      if (q == num_qubits - 1) {
        right_idx = right_boundary_index;
      } else {
        right_idx = internal_idx(q);
      }
      size_t d2 = dim(right_idx);

      std::vector<std::vector<std::vector<std::complex<double>>>> values =
        std::vector<std::vector<std::vector<std::complex<double>>>>(d1, 
                     std::vector<std::vector<std::complex<double>>>(d2, 
                                  std::vector<std::complex<double>>(2, std::complex<double>(0.0, 0.0))));
      for (uint32_t a1 = 0; a1 < d1; a1++) {
        for (uint32_t a2 = 0; a2 < d2; a2++) {
          for (uint32_t i = 0; i < 2; i++) {
            values[a1][a2][i] = eltC(T, left_idx=a1+1, right_idx=a2+1, ext=i+1);
          }
        }
      }

      return values;
    }

    double entanglement(uint32_t q, uint32_t index) {
      throw std::runtime_error("Have not yet implemented PauliMPS.entanglement");
    }

    std::complex<double> expectation(const PauliString& p) {
      throw std::runtime_error("Have not yet implemented PauliMPS.expectation(PauliString)");
      //if (p.num_qubits != num_qubits) {
      //  throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but PauliMPS has {} qubits.", p.num_qubits, num_qubits));
      //}

      //auto qubit_range = p.support_range();

      //// Pauli is proportional to I; return sign.
      //if (qubit_range == std::nullopt) {
      //  return p.sign();
      //}

      //auto [q1, q2] = qubit_range.value();

      //auto [L, R] = get_boundary_tensors(q1, q2);

      //std::vector<ITensor> paulis;
      //for (size_t q = q1; q < q2; q++) {
      //  Index idx = external_idx(q);
      //  paulis.push_back(pauli_tensor(p.to_pauli(q), prime(idx), idx));
      //}

      //return p.sign() * partial_expectation(paulis, q1, q2, L, R);
    }

    double expectation(const BitString& bits, std::optional<QubitSupport> support) {
      throw std::runtime_error("Have not yet implemented PauliMPS.expectation(BitString)");
      //QubitSupport _support;
      //if (support) {
      //  _support = support.value();
      //} else {
      //  _support = std::make_pair(0, num_qubits);
      //}

      //QubitInterval interval = to_interval(_support);

      //uint32_t q1, q2;
      //if (interval) {
      //  std::tie(q1, q2) = interval.value();
      //} else {
      //  return 1.0;
      //}

      //auto [L, R] = get_boundary_tensors(q1, q2);

      //std::vector<ITensor> operators;

      //Qubits qubits = to_qubits(_support);
      //std::set<uint32_t> qubits_set(qubits.begin(), qubits.end());
      //for (size_t q = q1; q < q2; q++) {
      //  Index idx = external_idx(q);
      //  ITensor op;
      //  if (qubits_set.contains(q)) {
      //    op = projection_tensor(bits.get(q), prime(idx), idx);
      //  } else {
      //    op = matrix_to_tensor(Eigen::Matrix2cd::Identity(), {prime(idx)}, {idx});
      //  }

      //  operators.push_back(op);
      //}

      //return partial_expectation(operators, q1, q2, L, R).real();
    }

    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) {
      throw std::runtime_error("Have not yet implemented PauliMPS.expectation(Matrix)");
      //if (qubits.size() == 0) {
      //  throw std::runtime_error("Cannot compute expectation over 0 qubits.");
      //}

      //size_t r = m.rows();
      //size_t c = m.cols();
      //size_t n = qubits.size();
      //if (r != c || (1u << n != r)) {
      //  throw std::runtime_error(fmt::format("Passed observable has dimension {}x{}, provided {} sites.", r, c, n));
      //}

      //if (!support_contiguous(qubits)) {
      //  throw std::runtime_error(fmt::format("Provided sites {} are not contiguous.", qubits));
      //}

      //auto [q1, q2] = to_interval(qubits).value();
      //
      //auto [L, R] = get_boundary_tensors(q1, q2);

      //return partial_expectation(m, q1, q2, L, R);
    }

    Eigen::VectorXcd coefficients() {
      throw std::runtime_error("Have not yet implemented PauliMPS.coefficients");
    }

    void evolve(const Eigen::Matrix2cd& gate, uint32_t q) {
      throw std::runtime_error("Have not yet implemented evolve on single qubit");
    }

    void evolve(const Eigen::Matrix4cd& gate, uint32_t q1_, uint32_t q2_) {
      throw std::runtime_error("Have not yet implemented evolve on two qubits");
    }

    void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
      assert_gate_shape(gate, qubits);

      if (qubits.size() == 1) {
        evolve(gate, qubits[0]);
        return;
      }

      // TODO combine q = 2 and q > 2 cases?
      if (qubits.size() == 2) {
        evolve(gate, qubits[0], qubits[1]);
        return;
      }

      throw std::runtime_error("Can only implement 1- and 2-qubit gates on PauliMPS.");
    }

    void reset_from_tensor(const ITensor& tensor, QubitInterval support) {
      if (!support) {
        return;
      }

      auto [q1, q2] = support.value();
      ITensor c = tensor;

      Index left = left_boundary_index;
      if (q1 != 0) {
        c *= singular_values[q1 - 1];
        left = internal_idx(q1 - 1);
      }
      
      Index right = right_boundary_index;
      if (q2 != num_qubits) {
        c *= singular_values[q2 - 1];
        right = internal_idx(q2 - 1);
      }

      for (size_t i = q1; i < q2 - 1; i++) {
        std::vector<Index> u_inds = {left, external_idx(i)};

        std::vector<Index> v_inds{right};
        for (size_t j = i + 1; j < q2; j++) {
          v_inds.push_back(external_idx(j));
        }
        
        auto [U, S, V] = svd(c, u_inds, v_inds, {
          "SVDMethod=","gesvd",
          "Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
          "LeftTags=",fmt::format("n={},Internal", i),
          "RightTags=",fmt::format("n={},Interal", i+1)
        });

        // Renormalize singular values
        size_t N = dim(inds(S)[0]);
        double d = 0.0;
        for (uint32_t p = 1; p <= N; p++) {
          std::vector<uint32_t> assignment = {p, p};
          double c = elt(S, assignment);
          d += c*c;
        }
        d = std::sqrt(d);
        S /= d;
        U *= d;

        c = V;
        tensors[i] = U;
        singular_values[i] = S;

        left = commonIndex(V, S);
        internal_indices[i]   = commonIndex(U, S);
        internal_indices[i+1] = commonIndex(U, S);
      }

      tensors[q2 - 1] = c;

      // Reset effect of multiplying in singular values
      auto inv = [&](Real r) { 
        if (r > sv_threshold) {
          return 1.0/r;
        } else {
          return 0.0;
        }
      };

      if (q1 != 0) {
        tensors[q1] *= apply(singular_values[q1 - 1], inv);
      }

      if (q2 != num_qubits) {
        tensors[q2 - 1] *= apply(singular_values[q2 - 1], inv);
      }
    }

    double trace() const {
      return 1.0;
    }

    bool measure(const Measurement& m) {
      throw std::runtime_error("Have not implemented PauliMPS.measure");
    }

    bool weak_measure(const WeakMeasurement& m) {
      throw std::runtime_error("Have not implemented PauliMPS.weak_measure");
    }
};


// ----------------------------------------------------------------------- //
// --------------- PauliMPS implementation --------------------- //
// ----------------------------------------------------------------------- //

PauliMPS::PauliMPS(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) : MagicQuantumState(num_qubits) {
  impl = std::make_unique<PauliMPSImpl>(num_qubits, bond_dimension, sv_threshold);
}

PauliMPS::PauliMPS(const PauliMPS& other) : MagicQuantumState(other.num_qubits) {
  impl = std::make_unique<PauliMPSImpl>(*other.impl.get());
}

PauliMPS::PauliMPS()=default;

PauliMPS::~PauliMPS()=default;

PauliMPS& PauliMPS::operator=(const PauliMPS& other) {
  if (this != &other) {
    impl = std::make_unique<PauliMPSImpl>(*other.impl);
  }
  return *this;
}

std::string PauliMPS::to_string() const {
  return impl->to_string();
}

double PauliMPS::entanglement(const QubitSupport& support, uint32_t index) {
  auto qubits = to_qubits(support);
	if (qubits.size() == 0) {
		return 0.0;
	}

	std::vector<uint32_t> sorted_qubits(qubits);
	std::sort(sorted_qubits.begin(), sorted_qubits.end());

  if (!support_contiguous(sorted_qubits)) {
    throw std::runtime_error(fmt::format("Cannot compute entanglement of non-contiguous qubits {} of MPS.", qubits));
  }

  uint32_t first = sorted_qubits[0];
  uint32_t last = sorted_qubits[sorted_qubits.size() - 1];
  if (first == 0) {
    return impl->entanglement(last + 1, index);
  } else if (last == num_qubits - 1) {
    return impl->entanglement(first, index);
  } else {
    throw std::runtime_error(fmt::format("Passed qubits {} do not share a boundary with an edge; cannot compute entanglement of MPS.", qubits));
  }
}

std::vector<std::vector<std::vector<std::complex<double>>>> PauliMPS::tensor(uint32_t q) const {
  return impl->get_tensor(q);
}

std::complex<double> PauliMPS::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

std::complex<double> PauliMPS::expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const {
  return impl->expectation(m, qubits);
}

double PauliMPS::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  return impl->expectation(bits, support);
}

std::shared_ptr<QuantumState> PauliMPS::partial_trace(const Qubits& qubits) const {
  throw std::runtime_error("Have not implemented PauliMPS.partial_trace");
}

Eigen::VectorXcd PauliMPS::coefficients() const {
  return impl->coefficients();
}

void PauliMPS::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
  impl->evolve(gate, qubit);
}

void PauliMPS::evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  impl->evolve(gate, qubits);
}

std::vector<double> PauliMPS::probabilities() const {
  Statevector psi(coefficients());
  return psi.probabilities();
}

double PauliMPS::purity() const {
  return 1.0;
}

bool PauliMPS::measure(const Measurement& m) {
  return impl->measure(m);
}

bool PauliMPS::weak_measure(const WeakMeasurement& m) {
  return impl->weak_measure(m);
}
