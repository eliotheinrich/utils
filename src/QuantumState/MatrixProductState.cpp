#include "QuantumStates.h"

#include <sstream>
#include <random>

#include <fmt/ranges.h>
#include <itensor/all.h>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

using namespace itensor;

static ITensor tensor_slice(const ITensor& tensor, const Index& index, int i) {
	if (!hasIndex(tensor, index)) {
		throw std::invalid_argument("Provided tensor cannot be sliced by provided index.");
	}

	auto v = ITensor(index);
	v.set(i, 1.0);

	return tensor*v;
}

static ITensor vector_to_tensor(const Eigen::VectorXcd& v, const std::vector<Index>& idxs) {
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

static ITensor matrix_to_tensor(
		const Eigen::Matrix2cd& matrix, 
		const Index i1, 
		const Index i2
	) {

	ITensor tensor(i1, i2);

	for (uint32_t i = 1; i <= 2; i++) {
		for (uint32_t j = 1; j <= 2; j++) {
			tensor.set(i1=i, i2=j, matrix(i-1,j-1));
		}
	}


	return tensor;
}

static ITensor matrix_to_tensor(
		const Eigen::Matrix4cd& matrix, 
		const Index i1, 
		const Index i2,
		const Index i3, 
		const Index i4
	) {

	ITensor tensor(i1, i2, i3, i4);

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

// This function is quite gross; cleanup?
static Index pad(ITensor& tensor, const Index& idx, uint32_t new_dim) {
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

	Index new_idx(new_dim, idx.tags());

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
	ITensor new_tensor(new_inds);

	for (const auto& it : iterInds(new_tensor)) {
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

static bool is_identity(const ITensor& I, double tol=1e-1) {
  auto idxs = inds(I);
  if (idxs.size() != 2) {
    return false;
  }

  Index i1 = idxs[0];
  Index i2 = idxs[1];

  if (dim(i1) != dim(i2)) {
    return false;
  }

  std::vector<uint32_t> idx_assignments(2);
  for (size_t k = 1; k < dim(i1) + 1; k++) {
    idx_assignments[0] = k;
    idx_assignments[1] = k;

    if (std::abs(eltC(I, idx_assignments) - 1.0) > tol) {
      return false;
    }
  }

  return true;
}

static Eigen::MatrixXcd tensor_to_matrix(const ITensor& A) {
  auto idxs = inds(A);
  if (idxs.size() != 2) {
    throw std::runtime_error("Can only convert order = 2 tensor to matrix.");
  }

  auto id1 = idxs[0];
  auto id2 = idxs[1];

  Eigen::MatrixXcd m(dim(id1), dim(id2));
  for (uint32_t i = 0; i < dim(id1); i++) {
    for (uint32_t j = 0; j < dim(id2); j++) {
      std::vector<uint32_t> assignments = {i + 1, j + 1};

      m(i, j) = eltC(A, assignments);
    }
  }

  return m;
}

static std::complex<double> tensor_to_scalar(const ITensor& A) {
  std::vector<uint32_t> assignments;
  return eltC(A, assignments);
}

static ITensor pauli_matrix(Pauli p, Index i1, Index i2) {
  if (p == Pauli::I) {
    return matrix_to_tensor(quantumstate_utils::I::value, i1, i2);
  } else if (p == Pauli::X) {
    return matrix_to_tensor(quantumstate_utils::X::value, i1, i2);
  } else if (p == Pauli::Y) {
    return matrix_to_tensor(quantumstate_utils::Y::value, i1, i2);
  } else if (p == Pauli::Z) {
    return matrix_to_tensor(quantumstate_utils::Z::value, i1, i2);
  }

  throw std::runtime_error("Invalid Pauli index.");
}


static void swap_tags(Index& idx1, Index& idx2) {
  IndexSet idxs = {idx1, idx2};
  auto new_idxs = swapTags(idxs, idx1.tags(), idx2.tags());
  idx1 = new_idxs[0];
  idx2 = new_idxs[1];
}

template <typename T1, typename T2, typename... Args>
static void swap_tags(Index& idx1, Index& idx2, T1& first, T2& second, Args&... args) {
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

template <typename... Tensors>
static void match_indices(const std::string& tags, ITensor& base, Tensors&... tensors) {
  Index idx = noPrime(findInds(base, tags)[0]);
  auto match = [&idx, &tags](ITensor& tensor) {
    Index idx_ = noPrime(findInds(tensor, tags)[0]);
    tensor.replaceInds({idx_}, {idx});
  };

  (match(tensors), ...);
}

static inline std::pair<uint32_t, uint32_t> get_qubit_range(const std::vector<uint32_t>& qubits) {
  return std::make_pair(static_cast<int>(*std::ranges::min_element(qubits)), static_cast<int>(*std::ranges::max_element(qubits)));
}

class MatrixProductStateImpl {
  friend class MatrixProductState;

  private:
		std::vector<itensor::ITensor> tensors;
		std::vector<itensor::ITensor> singular_values;
		std::vector<itensor::Index> external_indices;
		std::vector<itensor::Index> internal_indices;

    static const Eigen::Matrix2cd zero_projector() {
      return (Eigen::Matrix2cd() << 1, 0, 0, 0).finished();
    }

    static const Eigen::Matrix2cd one_projector() {
      return (Eigen::Matrix2cd() << 0, 0, 0, 1).finished();
    }

  public:
    uint32_t num_qubits;
    uint32_t bond_dimension;
    double sv_threshold;

    MatrixProductStateImpl()=default;
    ~MatrixProductStateImpl()=default;

    MatrixProductStateImpl(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) 
    : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold) {
      std::random_device random_device;

      if ((bond_dimension > 1u << num_qubits) && (num_qubits < 32)) {
        throw std::invalid_argument("Bond dimension must be smaller than 2^num_qubits.");
      }

      if (num_qubits <= 1) {
        throw std::invalid_argument("Number of qubits must be > 1 for MPS simulator.");
      }

      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        internal_indices.push_back(Index(1, fmt::format("Internal,Left,n={}", i)));
        internal_indices.push_back(Index(1, fmt::format("Internal,Right,n={}", i)));
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        external_indices.push_back(Index(2, fmt::format("External,i={}", i)));
      }

      ITensor tensor;

      tensor = ITensor(internal_indices[0], external_indices[0]);
      for (auto j : range1(internal_indices[0])) {
        tensor.set(j, 1, 1.0);
      }
      tensors.push_back(tensor);

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        tensor = ITensor(internal_indices[2*i], internal_indices[2*i + 1]);
        tensor.set(1, 1, 1);
        singular_values.push_back(tensor);

        if (i == num_qubits - 2) {
          tensor = ITensor(internal_indices[2*i + 1], external_indices[i+1]);
          for (auto j : range1(internal_indices[2*i + 1])) {
            tensor.set(j, 1, 1.0);
          }
        } else {
          tensor = ITensor(internal_indices[2*i + 1], internal_indices[2*i + 2], external_indices[i+1]);
          for (auto j1 : range1(internal_indices[2*i + 1])) {
            tensor.set(j1, j1, 1, 1.0);
          }
        }

        tensors.push_back(tensor);
      }
    }

    MatrixProductStateImpl(const MatrixProductStateImpl& other) : MatrixProductStateImpl(other.num_qubits, other.bond_dimension, other.sv_threshold) {
      tensors = other.tensors;
      singular_values = other.singular_values;

      internal_indices = other.internal_indices;
      external_indices = other.external_indices;
    }

    static MatrixProductStateImpl from_mps(const MPS& mps_, size_t bond_dimension, double sv_threshold) {
      size_t num_qubits = mps_.length();
      MatrixProductStateImpl vidal_mps(num_qubits, bond_dimension, sv_threshold);

      MPS mps(mps_);
      mps.position(0);

      ITensor U;
      ITensor V = mps(1);
      ITensor S;
      for (size_t i = 1; i < num_qubits; i++) {
        std::vector<Index> u_inds{siteIndex(mps, i)};
        std::vector<Index> v_inds{siteIndex(mps, i+1)};
        if (i != 1) {
          u_inds.push_back(vidal_mps.internal_idx(i - 2, InternalDir::Right));
        }

        if (i != mps.length() - 1) {
          v_inds.push_back(linkIndex(mps, i + 1));
        }


        auto M = V*mps(i + 1);

        std::tie(U, S, V) = svd(M, u_inds, v_inds,
            {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
             "LeftTags=",fmt::format("n={},Internal,Left",i-1),
             "RightTags=",fmt::format("n={},Internal,Right",i-1)});

        auto ext = siteIndex(mps, i);
        U.replaceTags(ext.tags(), vidal_mps.external_idx(i - 1).tags());
        if (i != 1) {
          U.replaceTags(linkIndex(mps, i-1).tags(), tags(vidal_mps.internal_idx(i-1, InternalDir::Left)));
        }

        if (i != num_qubits-1) {
          U.replaceTags(linkIndex(mps, i).tags(), tags(vidal_mps.internal_idx(i, InternalDir::Right)));
        }

        vidal_mps.tensors[i-1] = U;
        vidal_mps.singular_values[i-1] = S;

        vidal_mps.external_indices[i-1] = findInds(U, "External")[0];
        vidal_mps.internal_indices[2*(i-1)] = findInds(S, "Internal,Left")[0];
        vidal_mps.internal_indices[2*(i-1)+1] = findInds(S, "Internal,Right")[0];
      }

      V.replaceTags(siteIndex(mps, num_qubits).tags(), vidal_mps.external_idx(num_qubits - 1).tags());
      vidal_mps.external_indices[num_qubits - 1] = findInds(V, "External")[0];
      vidal_mps.tensors[num_qubits - 1] = V;

      return vidal_mps;
    }

    Index external_idx(size_t i) const {
      if (i >= num_qubits) {
        throw std::runtime_error(fmt::format("Cannot retrieve external index for i = {}.", i));
      }

      return external_indices[i];
    }

    enum InternalDir {
      Left, Right
    };

    Index internal_idx(size_t i, InternalDir d) const {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot retrieve internal index for i = {}.", i));
      }

      if (d == InternalDir::Left) {
        return internal_indices[2*i];
      } else {
        return internal_indices[2*i + 1];
      }
    }

    std::string to_string() const {
      Statevector sv(coefficients());
      return sv.to_string();
    }

    void print_mps(bool print_data=false) const {
      if (print_data) {
        PrintData(tensors[0]);
      } else {
        print(tensors[0]);
      }
      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        if (print_data) {
          PrintData(singular_values[i]);
        } else {
          print(singular_values[i]);
        }
        if (print_data) {
          PrintData(tensors[i+1]);
        } else {
          print(tensors[i+1]);
        }
      }
    }

    double entropy(uint32_t q, uint32_t index) {
      if (q < 0 || q > num_qubits) {
        throw std::invalid_argument("Invalid qubit passed to MatrixProductState.entropy; must have 0 <= q <= num_qubits.");
      }

      if (q == 0 || q == num_qubits) {
        return 0.0;
      }

      auto singular_vals = singular_values[q-1];
      int d = dim(inds(singular_vals)[0]);

      std::vector<double> sv(d);
      for (size_t i = 0; i < d; i++) {
        sv[i] = std::pow(elt(singular_vals, i+1, i+1), 2);
      }


      double s = 0.0;
      if (index == 1) {
        for (double v : sv) {
          if (v >= 1e-6) {
            s -= v * std::log2(v);
          }
        }
      } else {
        for (double v : sv) {
          s += std::pow(v, index);
        }
        
        s /= 1.0 - index;
      }

      return s;
    }

    ITensor A_l(size_t i) const {
      auto Ai = tensors[i];
      if (i != 0) {
        Ai *= singular_values[i - 1];
      }

      Index left;
      Index right;

      Index _left = (i > 0) ? internal_idx(i-1, InternalDir::Left) : Index();
      Index _right = (i < num_qubits - 1) ? internal_idx(i, InternalDir::Left) : Index();

      if (i == 0) {
        left = Index(1, "m=0,Internal");
        right = Index(dim(_right), "m=1,Internal");
        Ai.replaceInds({_right}, {right});

        ITensor one(left);
        one.set(left=1, 1.0);
        Ai *= one;
      } else if (i == num_qubits - 1) {
        left = Index(dim(_left), fmt::format("m={}, Internal", i));
        right = Index(1, fmt::format("m={}, Internal",i+1));
        Ai.replaceInds({_left}, {left});

        ITensor one(right);
        one.set(right=1, 1.0);
        Ai *= one;
      } else {
        left = Index(dim(_left), fmt::format("m={}, Internal", i));
        right = Index(dim(_right), fmt::format("m={}, Internal",i+1));
        Ai.replaceInds({_left, _right}, {left, right});
      }

      return Ai;
    }


    std::vector<ITensor> A_l(size_t q1_, size_t q2_) const {
      size_t q1 = std::min(q1_, q2_);
      size_t q2 = std::max(q1_, q2_);
      
      // Generate all tensors
      std::vector<ITensor> A;
      for (size_t i = q1; i < q2; i++) {
        A.push_back(A_l(i));
      }
      
      if (A.size() > 1) {
        // Align indices
        for (size_t i = 1; i < (q2 - q1); i++) {
          std::string s = fmt::format("m={}", i + q1);
          auto ai = findInds(A[i - 1], s);
          A[i].replaceInds(findInds(A[i], s), findInds(A[i - 1], s));
        }
      }

      return A;
    }

    ITensor A_r(size_t i) const {
      auto Ai = tensors[i];
      if (i != num_qubits - 1) {
        Ai *= singular_values[i];
      }

      Index left;
      Index right;

      Index _left = (i > 0) ? internal_idx(i-1, InternalDir::Right) : Index();
      Index _right = (i < num_qubits - 1) ? internal_idx(i, InternalDir::Right) : Index();

      if (i == 0) {
        left = Index(1, "m=0,Internal");
        right = Index(dim(_right), "m=1,Internal");
        Ai.replaceInds({_right}, {right});

        ITensor one(left);
        one.set(left=1, 1.0);
        Ai *= one;
      } else if (i == num_qubits - 1) {
        left = Index(dim(_left), fmt::format("m={},Internal", i));
        right = Index(1, fmt::format("m={},Internal",i+1));
        Ai.replaceInds({_left}, {left});

        ITensor one(right);
        one.set(right=1, 1.0);
        Ai *= one;
      } else {
        left = Index(dim(_left), fmt::format("m={},Internal", i));
        right = Index(dim(_right), fmt::format("m={},Internal",i+1));
        Ai.replaceInds({_left, _right}, {left, right});
      }

      return Ai;
    }

    std::vector<ITensor> A_r(size_t q1_, size_t q2_) const {
      size_t q1 = std::min(q1_, q2_);
      size_t q2 = std::max(q1_, q2_);
      
      // Generate all tensors
      std::vector<ITensor> A;
      for (size_t i = q1; i < q2; i++) {
        A.push_back(A_r(i));
      }
      
      if (A.size() > 1) {
        // Align indices
        for (size_t i = 1; i < (q2 - q1); i++) {
          std::string s = fmt::format("m={}", i + q1);
          auto ai = findInds(A[i - 1], s);
          A[i].replaceInds(findInds(A[i], s), findInds(A[i - 1], s));
        }
      }

      return A;
    }

    ITensor left_boundary_tensor(size_t q) const {
      Index left_idx;
      if (q == 0) {
        left_idx = Index(1, "m=0,Internal");
      } else {
        left_idx = Index(dim(internal_idx(q-1, InternalDir::Left)), fmt::format("m={},Internal", q));
      } 
      return delta(left_idx, prime(left_idx));
    }

    // Do not assume normalization holds; this is temporarily the case when performing batch measurements.
    ITensor left_environment_tensor(size_t q, const std::vector<ITensor>& external_tensors) const {
      if (q != external_tensors.size()) {
        throw std::runtime_error("Provided invalid number of external tensors to left_environment_tensor.");
      }

      ITensor L = left_boundary_tensor(0);

      extend_left_environment_tensor(L, 0, q, external_tensors);

      return L;
    }

    ITensor left_environment_tensor(size_t q) const {
      std::vector<ITensor> external_tensors;
      for (size_t i = 0; i < q; i++) {
        Index idx = external_idx(i);
        external_tensors.push_back(delta(idx, prime(idx)));
      }

      return left_environment_tensor(q, external_tensors);
    }

    void extend_left_environment_tensor(ITensor& L, uint32_t q1_, uint32_t q2_, const std::vector<ITensor>& external_tensors) const {
      uint32_t q1 = std::min(q1_, q2_);
      uint32_t q2 = std::max(q1_, q2_);
      if (q2 - q1 != external_tensors.size()) {
        throw std::runtime_error("Provided invalid number of external tensors to extend_left_environment_tensor.");
      }

      std::vector<ITensor> A = A_l(q1, q2);
      for (size_t j = q1; j < q2; j++) {
        ITensor Aj = A[j - q1];
        match_indices(fmt::format("m={}", j), L, Aj);
        L *= Aj;
        L *= external_tensors[j - q1];
        L *= conj(prime(Aj));
      }
    }

    void extend_left_environment_tensor(ITensor& L, uint32_t q1_, uint32_t q2_) const {
      uint32_t q1 = std::min(q1_, q2_);
      uint32_t q2 = std::max(q1_, q2_);
      std::vector<ITensor> external_tensors;
      for (size_t j = q1; j < q2; j++) {
        Index idx = external_idx(j);
        external_tensors.push_back(delta(idx, prime(idx)));
      }

      extend_left_environment_tensor(L, q1, q2, external_tensors);
    }

    ITensor right_boundary_tensor(size_t q) const {
      if (q == num_qubits - 1) {
        Index right_idx = Index(1, fmt::format("m={},Internal", num_qubits));
        return delta(right_idx, prime(right_idx));
      } else {
        // This is horrible and hacky but if singular_values is DiagReal, then ITensor throws an error contracting with itself
        // And if it is cast with toDense, it throws an error if it is already DenseReal.
        try {
          return singular_values[q] * conj(prime(singular_values[q], "Left"));
        } catch (const std::exception& e) {
          return toDense(singular_values[q]) * conj(prime(singular_values[q], "Left"));
        }
      } 
    }

    ITensor right_environment_tensor(size_t q, const std::vector<ITensor>& external_tensors) const {
      if (num_qubits - q != external_tensors.size()) {
        throw std::runtime_error("Provided invalid number of external tensors to right_environment_tensor.");
      }

      ITensor R = right_boundary_tensor(num_qubits - 1);

      extend_right_environment_tensor(R, num_qubits, q, external_tensors);

      return R;
    }

    ITensor right_environment_tensor(size_t q) const {
      std::vector<ITensor> external_tensors;
      for (size_t i = 0; i < q; i++) {
        Index idx = external_idx(num_qubits - i - 1);
        external_tensors.push_back(delta(idx, prime(idx)));
      }

      return right_environment_tensor(q, external_tensors);
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t q1_, uint32_t q2_, const std::vector<ITensor>& external_tensors) const {
      uint32_t q1 = std::max(q1_, q2_);
      uint32_t q2 = std::min(q1_, q2_);
      if (q1 - q2 != external_tensors.size()) {
        throw std::runtime_error("Provided invalid number of external tensors to extend_right_environment_tensor.");
      }

      std::vector<ITensor> A = A_l(q2+1, q1+1);
      for (size_t j = q1; j > q2; j--) {
        ITensor Aj = A[j - q2 - 1];
        match_indices(fmt::format("m={}", j + 1), R, Aj);
        R *= Aj;
        R *= external_tensors[q1 - j];
        R *= conj(prime(Aj));
      }
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t q1_, uint32_t q2_) const {
      uint32_t q1 = std::max(q1_, q2_);
      uint32_t q2 = std::min(q1_, q2_);

      std::vector<ITensor> external_tensors;
      for (size_t j = q1; j > q2; j--) {
        Index idx = external_idx(j);
        external_tensors.push_back(delta(idx, prime(idx)));
      }

      extend_right_environment_tensor(R, q1, q2, external_tensors);
    }

    std::complex<double> partial_expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits, const ITensor& L, const ITensor& R) const {
      ITensor contraction = L;
      if (qubits.size() == 1) {
        uint32_t q = qubits[0];
        Index ext = external_idx(q);
        auto A = A_l(q);
        ITensor mtensor = matrix_to_tensor(m, prime(ext), ext);

        match_indices(fmt::format("m={}", q), contraction, A);
        contraction *= A;
        contraction *= mtensor;
        contraction *= prime(conj(A));
      } else if (qubits.size() == 2) {
        auto [q1, q2] = get_qubit_range(qubits);
        Index ext1 = external_idx(q1);
        Index ext2 = external_idx(q2);
        ITensor mtensor = matrix_to_tensor(m, prime(ext1), prime(ext2), ext1, ext2);

        auto A1 = A_l(q1);
        auto A2 = A_l(q2);

        match_indices(fmt::format("m={}", q1), contraction, A1);
        match_indices(fmt::format("m={}", q2), A2, A1);

        contraction *= A1;
        contraction *= A2;
        contraction *= mtensor;
        contraction *= prime(conj(A1));
        contraction *= prime(conj(A2));
      } else {
        throw std::runtime_error(fmt::format("Provided qubits {} to partial_expectation, but {} qubit operations are not supported.", qubits, qubits.size()));
      }

      Index right_index = noPrime(inds(contraction)[0]);
      Index right_index_ = noPrime(inds(R)[0]);
      contraction *= delta(right_index, right_index_);
      contraction *= delta(prime(right_index), prime(right_index_));
      contraction *= R;
      return tensor_to_scalar(contraction);
    }

    double expectation(const PauliString& p) const {
      if (p.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but MatrixProductState has {} qubits.", p.num_qubits, num_qubits));
      }

      std::vector<ITensor> paulis(num_qubits);
      for (size_t i = 0; i < num_qubits; i++) {
        Index idx = external_indices[i];
        paulis[i] = pauli_matrix(p.to_pauli(i), idx, prime(idx));
      }

      ITensor C = tensors[0];
      ITensor contraction = C*paulis[0]*prime(conj(C));

      for (size_t i = 1; i < num_qubits; i++) {
        C = tensors[i]*singular_values[i-1];
        contraction *= C*paulis[i]*prime(conj(C));
      }

      double sign = p.get_r() ? -1.0 : 1.0;
      return sign*tensor_to_scalar(contraction).real();
    }

    std::complex<double> expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const {
      size_t r = m.rows();
      size_t c = m.cols();
      size_t n = sites.size();
      if (r != c || (1u << n != r)) {
        throw std::runtime_error(fmt::format("Passed observable has dimension {}x{}, provided {} sites.", r, c, n));
      }

      std::vector<uint32_t> sites_(sites.begin(), sites.end());
      std::sort(sites_.begin(), sites_.end());

      for (size_t i = 1; i < n; i++) {
        if (sites_[i] != sites_[i - 1] + 1) {
          throw std::runtime_error(fmt::format("Provided sites {} are not contiguous.", sites_));
        }
      }

      auto [q1, q2] = get_qubit_range(sites_);
      
      ITensor L = left_boundary_tensor(q1);
      ITensor R = right_boundary_tensor(q2);
      return partial_expectation(m, sites_, L, R);
    }

    std::vector<double> pauli_expectation_left_sweep(const PauliString& P, uint32_t q1_, uint32_t q2_) const {
      uint32_t q1 = std::min(q1_, q2_);
      uint32_t q2 = std::max(q1_, q2_);
      uint32_t subsystem_size = q2 - q1;
      if (subsystem_size == 0) {
        return {};
      }

      auto trace_tensor = [](const ITensor& T, const ITensor& R) {
        Index i = noPrime(inds(T)[0]);

        Index right_index = noPrime(inds(T)[0]);
        Index right_index_ = noPrime(inds(R)[0]);
        auto d1 = delta(right_index, right_index_);
        auto d2 = delta(prime(right_index), prime(right_index_));

        return tensor_to_scalar(T * delta(right_index, right_index_) * delta(prime(right_index), prime(right_index_)) * R).real();
      };

      std::vector<ITensor> paulis;
      auto add_paulis = [this, &P, &paulis](uint32_t i, uint32_t j) {
        for (size_t k = i; k < j; k++) {
          Index idx = external_idx(k);
          paulis.push_back(pauli_matrix(P.to_pauli(k), idx, prime(idx)));
        }
      };

      add_paulis(0, q1);

      ITensor L = left_boundary_tensor(0);
      extend_left_environment_tensor(L, 0, q1, paulis);
      ITensor R = right_boundary_tensor(q1);

      add_paulis(q1, q2);

      std::vector<uint32_t> qubits(q1);
      std::iota(qubits.begin(), qubits.end(), 0);

      std::vector<double> expectation;
      for (size_t i = q1; i < q2; i++) {
        extend_left_environment_tensor(L, i, i + 1, {paulis[i]});
        R = right_boundary_tensor(i);

        expectation.push_back(trace_tensor(L, R));

        qubits.push_back(i);
      }

      return expectation;
    }

    std::vector<double> pauli_expectation_right_sweep(const PauliString& P, uint32_t q1_, uint32_t q2_) const {
      uint32_t q1 = std::max(q1_, q2_);
      uint32_t q2 = std::min(q1_, q2_);
      uint32_t subsystem_size = q1 - q2;
      if (subsystem_size == 0) {
        return {};
      }

      auto trace_tensor = [](const ITensor& T, const ITensor& R) {
        Index i = noPrime(inds(T)[0]);

        Index right_index = noPrime(inds(T)[0]);
        Index right_index_ = noPrime(inds(R)[0]);
        auto d1 = delta(right_index, right_index_);
        auto d2 = delta(prime(right_index), prime(right_index_));

        return tensor_to_scalar(T * delta(right_index, right_index_) * delta(prime(right_index), prime(right_index_)) * R).real();
      };

      std::vector<ITensor> paulis;
      auto add_paulis = [this, &P, &paulis](uint32_t i, uint32_t j) {
        for (size_t k = i; k > j; k--) {
          Index idx = external_idx(k);
          paulis.push_back(pauli_matrix(P.to_pauli(k), idx, prime(idx)));
        }
      };

      add_paulis(num_qubits - 1, q1);

      ITensor L = toDense(left_boundary_tensor(q1));
      ITensor R = right_boundary_tensor(num_qubits - 1);
      extend_right_environment_tensor(R, num_qubits - 1, q1, paulis);

      add_paulis(q1, q2);

      std::vector<uint32_t> qubits(num_qubits - q1 - 1);
      std::iota(qubits.begin(), qubits.end(), q1 + 1);
      std::reverse(qubits.begin(), qubits.end());

      std::vector<double> expectation;
      size_t k = num_qubits - q1 - 1;
      for (size_t i = q1; i > q2; i--) {
        auto pauli = paulis[k];
        k++;
        extend_right_environment_tensor(R, i, i - 1, {pauli});
        L = toDense(left_boundary_tensor(i));
        expectation.push_back(trace_tensor(L, R));

        qubits.push_back(i);
      }

      return expectation;
    }


    std::pair<PauliString, double> sample_pauli(std::minstd_rand& rng) const {
      std::vector<Pauli> p(num_qubits);
      double P = 1.0;

      Index i(1, "i");
      Index j(1, "j");

      ITensor L(i, j);
      L.set(i=1, j=1, 1.0);

      for (size_t k = 0; k < num_qubits; k++) {
        std::vector<double> probs(4);
        std::vector<ITensor> tensors(4);

        auto Ak = A_r(k);
        std::string label1 = fmt::format("m={}", k);
        std::string label2 = fmt::format("m={}", k+1);
        Index alpha_left = findInds(Ak, label1)[0];
        L.replaceInds(inds(L), {alpha_left, prime(alpha_left)});

        Index s = external_indices[k];

        for (size_t p = 0; p < 4; p++) {
          auto sigma = pauli_matrix(static_cast<Pauli>(p), s, prime(s));

          auto C = prime(Ak)*conj(Ak)*sigma*L;
          auto contraction = conj(C)*C / 2.0;
          
          std::vector<size_t> inds;
          double prob = std::abs(eltC(contraction, inds));
          probs[p] = prob;
          tensors[p] = C / std::sqrt(2.0 * prob);
        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        size_t a = dist(rng);

        p[k] = static_cast<Pauli>(a);
        P *= probs[a];
        L = tensors[a];
      }

      double t = std::sqrt(P*std::pow(2.0, num_qubits));
      return std::make_pair(PauliString(p), t);
    }

    std::vector<PauliAmplitude> sample_paulis(size_t num_samples, std::minstd_rand& rng) const {
      std::vector<PauliAmplitude> samples(num_samples);

      for (size_t k = 0; k < num_samples; k++) {
        samples[k] = sample_pauli(rng);
      } 

      return samples;
    }


    ITensor coefficient_tensor() const {
      ITensor C = tensors[0];

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        C *= singular_values[i]*tensors[i+1];
      }

      return C;
    }

    std::complex<double> coefficients(size_t z) const {
      auto C = coefficient_tensor();

      std::vector<int> assignments(num_qubits);
      for (uint32_t j = 0; j < num_qubits; j++) {
        assignments[j] = ((z >> j) & 1u) + 1;
      }

      return eltC(C, assignments);
    }

    Eigen::VectorXcd coefficients() const {
      if (num_qubits > 31) {
        throw std::runtime_error("Cannot generate coefficients for n > 31 qubits.");
      }

      std::vector<uint32_t> indices(1u << num_qubits);
      std::iota(indices.begin(), indices.end(), 0);

      return coefficients(indices);
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

        vals[i] = eltC(C, assignments);
      }

      return vals;
    }

    void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
      auto i = external_indices[qubit];
      auto ip = prime(i);
      ITensor tensor = matrix_to_tensor(gate, ip, i);
      tensors[qubit] = noPrime(tensors[qubit]*tensor);
    }

    void swap(uint32_t q1, uint32_t q2) {
      evolve(quantumstate_utils::SWAP::value, {q1, q2});
    }

    void evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) {
      if (qubits.size() == 1) {
        evolve(gate, qubits[0]);
        return;
      }

      if ((qubits.size()) != 2 || (gate.rows() != gate.cols()) || (gate.rows() != (1u << qubits.size()))) {
        throw std::invalid_argument("Can only evolve two-qubit gates in MPS simulation.");
      }

      uint32_t q1 = std::min(qubits[0], qubits[1]);
      uint32_t q2 = std::max(qubits[0], qubits[1]);

      if (q2 - q1 > 1) {
        for (size_t k = q1; k < q2 - 1; k++) {
          swap(k, k+1);
        }

        std::vector<uint32_t> qbits{q2 - 1, q2};
        if (qubits[0] > qubits[1]) {
          Eigen::Matrix4cd SWAP = quantumstate_utils::SWAP::value;
          evolve(SWAP * gate * SWAP, qbits);
        } else {
          evolve(gate, qbits);
        }

        // Backward swaps
        for (size_t k = q2 - 1; k > q1; k--) {
          swap(k-1, k);
        }

        return;
      }

      auto i1 = external_indices[qubits[0]];
      auto i2 = external_indices[qubits[1]];
      ITensor gate_tensor = matrix_to_tensor(gate, 
        prime(i1), prime(i2), 
        i1, i2
      );

      ITensor theta = tensors[q1];
      theta *= singular_values[q1];
      theta *= tensors[q2];
      theta *= gate_tensor;
      theta = noPrime(theta);

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

      auto [U, D, V] = svd(theta, u_inds, v_inds, 
          {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
           "LeftTags=",fmt::format("Internal,Left,n={}", q1),
           "RightTags=",fmt::format("Internal,Right,n={}", q1)});

      internal_indices[2*q1] = commonIndex(U, D);
      internal_indices[2*q2 - 1] = commonIndex(V, D);


      auto inv = [](Real r) { return 1.0/r; };
      if (q1 != 0) {
        U *= apply(singular_values[q1-1], inv);
      }
      if (q2 != num_qubits - 1) {
        V *= apply(singular_values[q2], inv);
      }

      tensors[q1] = U;
      tensors[q2] = V;
      singular_values[q1] = D;
    }

    void reset_from_tensor(const ITensor& tensor) {
      ITensor c = tensor;
      for (size_t i = 0; i < num_qubits - 1; i++) {
        std::vector<Index> u_inds = {external_idx(i)};
        if (i > 0) {
          u_inds.push_back(internal_idx(i-1, InternalDir::Right));
        }

        std::vector<Index> v_inds;
        for (size_t j = i + 1; j < num_qubits; j++) {
          v_inds.push_back(external_idx(j));
        }
        
        auto [U, D, V] = svd(c, u_inds, v_inds,
            {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
            "LeftTags=",fmt::format("Internal,Left,n={}", i),
            "RightTags=",fmt::format("Internal,Right,n={}", i)});

        c = V;
        tensors[i] = U;
        singular_values[i] = D;

        internal_indices[2*i] = commonIndex(U, D);
        internal_indices[2*i + 1] = commonIndex(V, D);
      }

      tensors[num_qubits - 1] = c;
    }

    void id(uint32_t q1, uint32_t q2) {
      evolve(Eigen::Matrix4cd::Identity(), {q1, q2});
    }

    double trace() const {
      return inner(*this).real();
    }

    size_t bond_dimension_at_site(size_t i) const {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot check bond dimension of site {} for MPS with {} sites.", i, num_qubits));
      }

      return dim(inds(singular_values[i])[0]);
    }

    void reverse() {
      for (size_t i = 0; i < num_qubits/2; i++) {
        size_t j = num_qubits - i - 1;
        std::swap(tensors[i], tensors[j]);
      }

      size_t k = (num_qubits % 2) ? (num_qubits / 2) : (num_qubits / 2 - 1);
      for (size_t i = 0; i < k; i++) {
        size_t j = num_qubits - i - 2;
        std::swap(singular_values[i], singular_values[j]);
      }


      // Relabel indices
      for (size_t i = 0; i < num_qubits/2; i++) {
        size_t j = num_qubits - i - 1;
        std::swap(external_indices[i], external_indices[j]);
        swap_tags(external_indices[i], external_indices[j], tensors[i], tensors[j]);
      }

      for (size_t i = 0; i < num_qubits - 1; i++) {
        size_t j = internal_indices.size() - i - 1;
        std::swap(internal_indices[i], internal_indices[j]);
      }

      for (size_t i = 0; i < k; i++) {
        size_t j = internal_indices.size() - i - 1;
        size_t k1 = i / 2;
        size_t k2 = num_qubits - k1 - 1;

        if (i % 2 == 0) {
          swap_tags(internal_indices[i], internal_indices[j], singular_values[k1], singular_values[k2 - 1], tensors[k1], tensors[k2]);
        } else {
          swap_tags(internal_indices[i], internal_indices[j], singular_values[k1], singular_values[k2 - 1], tensors[k1 + 1], tensors[k2 - 1]);
        }
      }
    }

    std::complex<double> inner(const MatrixProductStateImpl& other) const {
      if (num_qubits != other.num_qubits) {
        throw std::runtime_error("Can't compute inner product of MPS; number of qubits do not match.");
      }
      
      Index ext1 = external_idx(0);
      Index ext2 = other.external_idx(0);
      ITensor contraction = tensors[0] * prime(conj(replaceInds(other.tensors[0], {ext2}, {ext1})), "Internal");

      for (size_t i = 1; i < num_qubits; i++) {
        ext1 = external_idx(i);
        ext2 = other.external_idx(i);
        contraction *= singular_values[i - 1];
        contraction *= tensors[i];
        contraction *= prime(other.singular_values[i - 1]);
        contraction *= prime(conj(replaceInds(other.tensors[i], {ext2}, {ext1})), "Internal");
      }

      return std::conj(tensor_to_scalar(contraction));
    }

    bool singular_values_trivial(size_t i) const {
      return (dim(internal_idx(i, InternalDir::Right)) == 1)
          && (std::abs(norm(singular_values[i]) - 1.0) < 1e-4);
    }

    size_t normalize(uint32_t q1, uint32_t q2, uint32_t lq, uint32_t rq) {
      size_t num_svd = 0;
      for (uint32_t i = q2; i < rq; i++) {
        if (singular_values_trivial(i)) {
          break;
        }

        id(i, i+1);
        num_svd++;
      }

      for (uint32_t i = q1; i > lq; i--) {
        if (singular_values_trivial(i - 1)) {
          break;
        }

        id(i-1, i);
        num_svd++;
      }

      return num_svd;
    }

    MeasurementOutcome measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits, double r, const ITensor& L, const ITensor& R) const {
      auto pm = p.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << p.num_qubits, 1u << p.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;
      Eigen::MatrixXcd proj1 = (id - pm)/2.0;

      double prob_zero = std::abs(partial_expectation(proj0, qubits, L, R));

      bool outcome = r > prob_zero;

      auto proj = outcome ? proj1 / std::sqrt(1.0 - prob_zero) : proj0 / std::sqrt(prob_zero);

      return {proj, prob_zero, outcome};
    }

    MeasurementOutcome measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits, double r) const {
      auto [q1, q2] = get_qubit_range(qubits);
      ITensor L = left_boundary_tensor(q1);
      ITensor R = right_boundary_tensor(q2);

      return measurement_outcome(p, qubits, r, L, R);
    }

    void apply_measure(const MeasurementOutcome& outcome, const std::vector<uint32_t>& qubits, bool renormalize) {
      auto proj = std::get<0>(outcome);
      evolve(proj, qubits);
      if (renormalize) {
        uint32_t q1 = *std::ranges::min_element(qubits);
        uint32_t q2 = *std::ranges::max_element(qubits);
        normalize(q1, q2, 0, num_qubits - 1);
      }
    }

    template <typename T>
    std::vector<T> sort_measurements(const std::vector<T>& measurements) {
      for (const auto& m : measurements) {
        auto [q1, q2] = get_qubit_range(std::get<1>(m));
        if (std::abs(static_cast<int>(q1) - static_cast<int>(q2)) > 1) {
          throw std::runtime_error(fmt::format("Qubits = {} are not adjacent and thus cannot be measured on MPS.", std::get<1>(m)));
        }
      }

      std::vector<T> sorted_measurements = measurements;

      std::sort(sorted_measurements.begin(), sorted_measurements.end(), [](const T& m1, const T& m2) {
        auto [a1, a2] = get_qubit_range(std::get<1>(m1));
        auto [b1, b2] = get_qubit_range(std::get<1>(m2));
        return a1 < b1;
      });

      return sorted_measurements;
    }

    template <typename T>
    std::vector<uint32_t> get_next_qubits(const std::vector<T>& measurements) {
      size_t num_measurements = measurements.size();
      std::vector<bool> mask(num_qubits, false);
      size_t right_qubit = 0;
      for (auto const& m : measurements) {
        auto qubits = std::get<1>(m);
        for (auto q : qubits) {
          if (q > right_qubit) {
            right_qubit = q;
          }
          mask[q] = true;
        }
      }

      std::vector<uint32_t> next_qubit;
      for (size_t i = 0; i < num_measurements - 1; i++) {
        auto qubits = std::get<1>(measurements[i]);
        auto [q1, q2] = get_qubit_range(qubits);
        size_t n = q2 + 1;
        while (n < num_qubits) {
          if (mask[n]) {
            next_qubit.push_back(n);
            break;
          }
          n++;
        }
      }

      return next_qubit;
    }
    
    std::vector<bool> measure(const std::vector<MeasurementData>& measurements, const std::vector<double>& random_vals) {
      auto sorted_measurements = sort_measurements(measurements);
      size_t num_measurements = sorted_measurements.size();
      if (num_measurements == 0) {
        return {};
      };

      size_t left_qubit = get_qubit_range(std::get<1>(sorted_measurements[0])).first;
      Index left_index = findIndex(A_l(left_qubit), fmt::format("m={}", left_qubit));
      ITensor L = left_boundary_tensor(left_qubit);

      std::vector<bool> results;
      for (size_t i = 0; i < num_measurements; i++) {
        const auto& [p, qubits] = sorted_measurements[i];
        auto [q1, q2] = get_qubit_range(qubits);
        std::vector<ITensor> A = A_l(left_qubit, q1+1);
        for (size_t j = left_qubit; j < q1; j++) {
          ITensor Aj = A[j - left_qubit];
          match_indices(fmt::format("m={}", j), L, Aj);
          L *= Aj;
          L *= conj(prime(Aj, "Internal"));
        }
        left_qubit = q1;

        ITensor R = right_boundary_tensor(q2);
        auto outcome = measurement_outcome(p, qubits, random_vals[i], L, R);
        apply_measure(outcome, qubits, false);
        results.push_back(std::get<2>(outcome));
      }

      for (size_t i = num_qubits - 1; i > 0; i--) {
        uint32_t j = i - 1;
        uint32_t k = i;

        id(j, k);
      }

      for (size_t i = 0; i < num_qubits - 1; i++) {
        uint32_t j = i;
        uint32_t k = i + 1;

        id(j, k);
      }

      return results;
    }

    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits, double r) {
      if (qubits.size() != p.num_qubits) {
        throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
      }

      auto outcome = measurement_outcome(p, qubits, r);
      apply_measure(outcome, qubits, true);
      
      return std::get<2>(outcome);
    }

    bool measure(uint32_t q, double r) {
      return measure(PauliString("Z"), {q}, r);
    }

    MeasurementOutcome weak_measurement_outcome(const PauliString&p, const std::vector<uint32_t>& qubits, double beta, double r, const ITensor& L, const ITensor& R) const {
      auto pm = p.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << p.num_qubits, 1u << p.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;

      double prob_zero = std::abs(partial_expectation(proj0, qubits, L, R));

      bool outcome = r >= prob_zero;

      Eigen::MatrixXcd t = pm;
      if (outcome) {
        t = -t;
      }

      Eigen::MatrixXcd proj = (beta*t).exp();

      Eigen::MatrixXcd P = proj.pow(2);
      double norm = std::sqrt(std::abs(partial_expectation(P, qubits, L, R)));

      proj = proj / norm;

      return {proj, prob_zero, outcome};
    }

    MeasurementOutcome weak_measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits, double beta, double r) const {
      auto [q1, q2] = get_qubit_range(qubits);
      ITensor L = left_boundary_tensor(q1);
      ITensor R = right_boundary_tensor(q2);

      return weak_measurement_outcome(p, qubits, beta, r, L, R);
    }

    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta, double r) {
      if (qubits.size() != p.num_qubits) {
        throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
      }

      auto outcome = weak_measurement_outcome(p, qubits, beta, r);
      apply_measure(outcome, qubits, true);
      
      return std::get<2>(outcome);
    }

    std::vector<bool> weak_measure(const std::vector<WeakMeasurementData>& measurements, const std::vector<double>& random_vals) {
      auto sorted_measurements = sort_measurements(measurements);
      size_t num_measurements = sorted_measurements.size();
      if (num_measurements == 0) {
        return {};
      };

      size_t left_qubit = get_qubit_range(std::get<1>(sorted_measurements[0])).first;
      Index left_index = findIndex(A_r(left_qubit), fmt::format("m={}", left_qubit));
      ITensor L = left_boundary_tensor(left_qubit);

      std::vector<bool> results;
      for (size_t i = 0; i < num_measurements; i++) {
        const auto& [p, qubits, beta] = sorted_measurements[i];
        auto [q1, q2] = get_qubit_range(qubits);
        extend_left_environment_tensor(L, left_qubit, q1);
        left_qubit = q1;

        ITensor R = right_boundary_tensor(q2);
        auto outcome = weak_measurement_outcome(p, qubits, beta, random_vals[i], L, R);
        apply_measure(outcome, qubits, false);
        results.push_back(std::get<2>(outcome));
      }

      extend_left_environment_tensor(L, left_qubit, num_qubits);
      L *= delta(inds(L));
      double tr = tensor_to_scalar(L).real();

      double c = std::pow(tr, 1.0/(4.0*(num_qubits - 1.0)));

      for (size_t i = num_qubits - 1; i > 0; i--) {
        uint32_t j = i - 1;
        uint32_t k = i;

        evolve(Eigen::Matrix4cd::Identity() / c, {j, k});
      }

      for (size_t i = 0; i < num_qubits - 1; i++) {
        uint32_t j = i;
        uint32_t k = i + 1;

        evolve(Eigen::Matrix4cd::Identity() / c, {j, k});
      }

      return results;
    }

    // ======================================= DEBUG FUNCTIONS ======================================= //
    ITensor orthogonality_tensor_r(uint32_t i) const {
      ITensor A = A_r(i);
      return A * conj(prime(A, fmt::format("m={}",i)));
    }

    ITensor orthogonality_tensor_l(uint32_t i) const {
      auto A = A_l(i);
      return A * conj(prime(A, fmt::format("m={}", i+1)));
    }

    std::vector<size_t> orthogonal_sites_r() const {
      std::vector<size_t> sites;
      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_r(i);
        if (!is_identity(I)) {
          sites.push_back(i);
        }
      }

      return sites;
    }

    std::vector<size_t> orthogonal_sites_l() const {
      std::vector<size_t> sites;
      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_l(i);
        if (!is_identity(I)) {
          sites.push_back(i);
        }
      }

      return sites;
    }

    std::vector<size_t> normalization_sites() const {
      std::vector<size_t> sites;
      for (size_t i = 0; i < num_qubits-1; i++) {
        size_t d = dim(inds(singular_values[i])[0]);
        for (size_t j = 1; j <= d; j++) {
          double v = elt(singular_values[i], j, j);
          if (v > 1.0 + 1e-5 || std::isnan(v)) {
            sites.push_back(i);
          }
        }
      }

      return sites;
    }

    void show_problem_sites() const {
      auto ortho_sites_l = orthogonal_sites_l();
      auto ortho_sites_r = orthogonal_sites_r();
      auto normo_sites = normalization_sites();

      std::cout << fmt::format("ortho sites_l: {}\n", ortho_sites_l);
      std::cout << fmt::format("ortho sites_r: {}\n", ortho_sites_r);
      std::cout << fmt::format("normo sites: {}\n", normo_sites);
    }

    bool state_valid() const {
      for (size_t i = 0; i < num_qubits - 1; i++) {
        size_t d = dim(inds(singular_values[i])[0]);
        for (size_t j = 1; j <= d; j++) {
          double v = elt(singular_values[i], j, j);
          if (v > 1.0 + 1e-5 || std::isnan(v)) {
            return false;
          }
        }
      }

      return true;
    }

    bool check_orthonormality() const {
      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_l(i);
        if (!is_identity(I)) {
          PrintData(I);
          return false;
        }
      }

      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_r(i);
        if (!is_identity(I)) {
          PrintData(I);
          return false;
        }
      }

      return true;
    }

    auto get_mask_right() const {
      std::vector<int> mask(num_qubits);
      for (size_t k = 0; k < num_qubits; k++) {
        auto I = orthogonality_tensor_r(k);
        mask[k] = is_identity(I);
      }
      return mask;
    }

    auto get_mask_left() const {
      std::vector<int> mask(num_qubits);
      for (size_t k = 0; k < num_qubits; k++) {
        auto I = orthogonality_tensor_l(k);
        mask[k] = is_identity(I);
      }
      return mask;
    }

    void print_mask(const std::vector<int>& mask) const {
      std::cout << "| ";
      for (size_t k = 0; k < num_qubits; k++) {
        std::cout << mask[k] << " ";
      }
      std::cout << "|\n";
    }

    void display_masks(int i, int j, const std::vector<int>& mask_left1, const std::vector<int>& mask_left2, 
                                     const std::vector<int>& mask_right1, const std::vector<int>& mask_right2) {
      std::cout << "       | ";
      for (size_t k = 0; k < num_qubits; k++) {
        if (k == i || k == j) {
          std::cout << "v ";
        } else {
          std::cout << "  ";
        }
      }
      std::cout << "|\n";
      std::cout << "Left:  ";
      print_mask(mask_left1);
      std::cout << "Left:  ";
      print_mask(mask_left2);
      std::cout << "Right: ";
      print_mask(mask_right1);
      std::cout << "Right: ";
      print_mask(mask_right2);
    }

    void id_debug(uint32_t i1, uint32_t i2, double d = 1.0) {
      auto mask_left1 = get_mask_left();
      auto mask_right1 = get_mask_right();
      evolve(Eigen::Matrix4cd::Identity(), {i1, i2});
      auto mask_left2 = get_mask_left();
      auto mask_right2 = get_mask_right();
      display_masks(i1, i2, mask_left1, mask_left2, mask_right1, mask_right2);
    }
    // ======================================= DEBUG FUNCTIONS ======================================= //
};

class MatrixProductOperatorImpl {
  friend class MatrixProductOperator;

  public:
    using MPOBlock = std::optional<ITensor>;

    size_t num_qubits;
    uint32_t bond_dimension;
    double sv_threshold;

    ITensor left_block;
    std::vector<ITensor> ops;
    std::vector<MPOBlock> blocks;
    std::vector<Index> internal_indices;
    std::vector<Index> external_indices;

    static ITensor get_block_left(const std::vector<ITensor>& A, size_t q, const Index& i_r) {
      if (q == 0) {
        // TODO see if toDense can be removed
        return toDense(delta(i_r, prime(i_r)));
      }

      auto Ak = A[0];
      auto a0 = findIndex(Ak, "m=0");
      auto C = Ak*conj(prime(Ak, "Internal"))*delta(a0, prime(a0));
      for (size_t k = 1; k < q; k++) {
        Ak = A[k];
        C *= Ak*conj(prime(Ak, "Internal"));
      }

      Index i = noPrime(findInds(C, fmt::format("m={}", q))[0]);

      C.replaceInds({i, prime(i)}, {i_r, prime(i_r)});
      return C;
    }

    static MPOBlock get_block_right(const std::vector<ITensor>& A, size_t q, const Index& i_l) {
      size_t L = A.size();
      if (q == L - 1) {
        // TODO see if toDense can be removed
        return toDense(delta(i_l, prime(i_l)));
      }

      auto Ak = A[L - 1];
      auto aL = findIndex(Ak, fmt::format("m={}", L));
      auto C = Ak*conj(prime(Ak, "Internal"))*delta(aL, prime(aL));

      for (size_t k = L - 2; k > q; k--) {
        Ak = A[k];
        C *= Ak*conj(prime(Ak, "Internal"));
      }

      Index i = noPrime(findInds(C, fmt::format("m={}", q+1))[0]);

      C.replaceInds({i, prime(i)}, {i_l, prime(i_l)});
      return C;
    }

    static MPOBlock get_block(const std::vector<ITensor>& A, size_t q1, size_t q2, const Index& i_l, const Index& i_r) {
      if (q2 == q1 + 1) {
        return std::nullopt;
      }

      auto Ak = A[q1 + 1];
      auto C = Ak*conj(prime(Ak, "Internal"));
      for (size_t k = q1 + 2; k < q2; k++) {
        Ak = A[k];
        C *= Ak*conj(prime(Ak, "Internal"));
      }

      Index i1 = noPrime(findInds(C, fmt::format("m={}", q1+1))[0]);
      Index i2 = noPrime(findInds(C, fmt::format("m={}", q2))[0]);

      C.replaceInds({i1, prime(i1), i2, prime(i2)}, {i_l, prime(i_l), i_r, prime(i_r)});
      return C;
    }

    enum InternalDir {
      Left, Right
    };

    Index internal_idx(size_t i, InternalDir dir) const {
      if (dir == InternalDir::Left) {
        return internal_indices[2*i];
      } else {
        return internal_indices[2*i + 1];
      } 
    }

    MatrixProductOperatorImpl()=default;

    MatrixProductOperatorImpl(const MatrixProductOperatorImpl& other) {
      num_qubits = other.num_qubits;
      bond_dimension = other.bond_dimension;
      sv_threshold = other.sv_threshold;

      left_block = other.left_block;
      ops = other.ops;
      blocks = other.blocks;

      internal_indices = other.internal_indices;
      external_indices = other.external_indices;
    }

    MatrixProductOperatorImpl(const MatrixProductStateImpl& mps, const std::vector<uint32_t>& traced_qubits)
      : num_qubits(mps.num_qubits - traced_qubits.size()), bond_dimension(mps.bond_dimension), sv_threshold(mps.sv_threshold) {
      if (traced_qubits.size() >= mps.num_qubits) {
        throw std::runtime_error(fmt::format("Passed {} qubits to trace over an MatrixProductState with {} qubits. Must be at least one remaining physical qubit.", traced_qubits.size(), mps.num_qubits));
      }

      std::vector<bool> mask(mps.num_qubits, false);
      for (auto const q : traced_qubits) {
        mask[q] = true;
      }

      // Generate all tensors
      std::vector<ITensor> A = mps.A_r(0, mps.num_qubits);

      std::vector<uint32_t> external_qubits;
      size_t k = 0;
      for (size_t i = 0; i < mps.num_qubits; i++) {
        if (!mask[i]) {
          external_qubits.push_back(i);
          auto Ai = A[i];
          Index external_idx = findIndex(Ai, "External");
          Index external_idx_(dim(external_idx), fmt::format("i={},External",k));

          Index internal_idx1 = findIndex(Ai, fmt::format("m={}", i));
          Index internal_idx2 = findIndex(Ai, fmt::format("m={}", i+1));
          Index internal_idx1_ = Index(dim(internal_idx1), fmt::format("n={},Internal,Left", k));
          Index internal_idx2_ = Index(dim(internal_idx2), fmt::format("n={},Internal,Right", k));

          ops.push_back(replaceInds(Ai, {external_idx, internal_idx1, internal_idx2}, {external_idx_, internal_idx1_, internal_idx2_}));

          external_indices.push_back(external_idx_);
          internal_indices.push_back(internal_idx1_);
          internal_indices.push_back(internal_idx2_);
          k++;
        }
      }

      blocks = std::vector<MPOBlock>();
      left_block = get_block_left(A, external_qubits[0], internal_idx(0, InternalDir::Left));

      for (size_t i = 0; i < num_qubits - 1; i++) {
        size_t q1 = external_qubits[i];
        size_t q2 = external_qubits[i + 1];

        blocks.push_back(get_block(A, q1, q2, internal_idx(i, InternalDir::Right), internal_idx(i + 1, InternalDir::Left)));
      }

      blocks.push_back(get_block_right(A, external_qubits[num_qubits - 1], internal_idx(num_qubits - 1, InternalDir::Right)));
    }

    MatrixProductOperatorImpl(const MatrixProductOperatorImpl& mpo, const std::vector<uint32_t>& traced_qubits)
      : num_qubits(mpo.num_qubits - traced_qubits.size()), bond_dimension(mpo.bond_dimension), sv_threshold(mpo.sv_threshold) {
      if (traced_qubits.size() >= mpo.num_qubits) {
        throw std::runtime_error(fmt::format("Passed {} qubits to trace over an MatrixProductOperator with {} qubits. Must be at least one remaining physical qubit.", traced_qubits.size(), mpo.num_qubits));
      }

      std::vector<uint32_t> sorted_qubits(traced_qubits.begin(), traced_qubits.end());
      std::sort(sorted_qubits.begin(), sorted_qubits.end());

      size_t num_traced_qubits = traced_qubits.size();

      // Mask of physical qubits vs traced qubits
      std::vector<bool> mask(mpo.num_qubits, true);
      for (auto q : sorted_qubits) {
        mask[q] = false;
      }

      std::vector<uint32_t> physical_qubits;
      for (size_t i = 0; i < mpo.num_qubits; i++) {
        if (mask[i]) {
          physical_qubits.push_back(i);
        }
      }

      std::vector<ITensor> deltas;
      for (size_t i = 0; i < physical_qubits[0]; i++) {
        Index e = mpo.external_idx(i);
        deltas.push_back(delta(e, prime(e)));
      }

      if (physical_qubits[0] == 0) {
        left_block = mpo.left_block;
      } else {
        left_block = mpo.partial_contraction(0, physical_qubits[0], deltas);
      }

      for (size_t k = 0; k < physical_qubits.size(); k++) {
        uint32_t q = physical_qubits[k];
        ITensor new_op = mpo.ops[q];
        Index i1 = findIndex(new_op, "External");
        Index i2 = Index(dim(i1), fmt::format("i={},External", k));
        new_op.replaceInds({i1}, {i2}); 
        external_indices.push_back(i2);
        ops.push_back(new_op);


        if (q == mpo.num_qubits - 1 || mask[q + 1]) {
          // If end of chain or next block is physical, next block is unchanged
          blocks.push_back(mpo.blocks[q]);
        } else { 
          // Otherwise, next qubit is traced over; perform partial contraction
          size_t q1 = q + 1;
          size_t q2 = q1;
          while (!mask[q2] && q2 < mpo.num_qubits) {
            q2++;
          }

          deltas = std::vector<ITensor>();
          for (size_t i = q1; i < q2; i++) {
            Index e = mpo.external_idx(i);
            deltas.push_back(delta(e, prime(e)));
          }

          auto new_block = mpo.partial_contraction(q1, q2, deltas);
          
          blocks.push_back(mpo.partial_contraction(q1, q2, deltas));
        }
      }

      for (size_t i = 0; i < ops.size(); i++) {
        Index aL = noPrime(findInds(ops[i], "Internal,Left")[0]);
        Index aR = noPrime(findInds(ops[i], "Internal,Right")[0]);
        Index aL_ = Index(dim(aL), fmt::format("n={},Internal,Left", i));
        Index aR_ = Index(dim(aR), fmt::format("n={},Internal,Right", i));

        ops[i].replaceInds({aL, aR}, {aL_, aR_});

        if (i == 0) {
          // TODO see if toDense can be removed
          left_block.replaceInds({aL, prime(aL)}, {aL_, prime(aL_)});
        } else {
          if (blocks[i - 1]) {
            blocks[i - 1] = blocks[i - 1].value() * delta(aL, aL_) * delta(prime(aL), prime(aL_));
          }
        }

        if (i == ops.size() - 1) {
          blocks[i] = toDense(delta(aR_, prime(aR_)));
        } else if (blocks[i]) {
          blocks[i] = blocks[i].value().replaceInds({aR, prime(aR)}, {aR_, prime(aR_)});
        }

        internal_indices.push_back(aL_);
        internal_indices.push_back(aR_);
      }
    }

    void print_mps() const {
      std::cout << "left_block: " << block_to_string(left_block) << "\n";
      for (size_t i = 0; i < num_qubits; i++) {
        std::cout << fmt::format("ops[{}] = ", i);
        std::cout << ops[i] << "\n";
        std::cout << fmt::format("blocks[{}] = ", i);
        std::cout << block_to_string(blocks[i]) << "\n";
      }
    }

    Index external_idx(size_t i) const {
      return findInds(ops[i], "External")[0];
    }

    ITensor apply_block_r(const ITensor& tensor, size_t i) const {
      if (blocks[i].has_value()) {
        return tensor*blocks[i].value();
      } else {
        Index i1 = internal_idx(i, InternalDir::Right);
        Index i2 = internal_idx(i+1, InternalDir::Left);
        return replaceInds(tensor, {i1, prime(i1)}, {i2, prime(i2)});
      }
    }

    ITensor apply_block_l(const ITensor& tensor, size_t i) const {
      if (i == 0) {
        return tensor*left_block;
      } else if (blocks[i-1].has_value()) {
        return tensor*blocks[i-1].value();
      } else {
        Index i1 = internal_idx(i, InternalDir::Left);
        Index i2 = internal_idx(i-1, InternalDir::Right);

        return replaceInds(tensor, {i1, prime(i1)}, {i2, prime(i2)});
      }
    }

    bool block_trivial(size_t k) const {
      if (k < num_qubits - 1) {
        return !blocks[k].has_value();
      }

      return is_identity(blocks[k].value());
    }

    double trace() const {
      std::vector<ITensor> deltas;
      for (size_t i = 0; i < num_qubits; i++) {
        Index idx = external_idx(i);
        deltas.push_back(delta(idx, prime(idx)));
      }

      auto t = partial_contraction(0, num_qubits, deltas);
      return tensor_to_scalar(t).real();
    }

    Eigen::MatrixXcd coefficients() const {
      if (num_qubits > 31) {
        throw std::runtime_error("Cannot generate coefficients for n > 31 qubits.");
      }

      auto contraction = partial_contraction(0, num_qubits);

      size_t s = 1u << num_qubits; 
      Eigen::MatrixXcd data = Eigen::MatrixXcd::Zero(s, s);

      for (size_t z1 = 0; z1 < s; z1++) {
        for (size_t z2 = 0; z2 < s; z2++) {
          std::vector<int> assignments(2*num_qubits);
          for (size_t j = 0; j < num_qubits; j++) {
            assignments[2*j + 1] = ((z1 >> j) & 1u) + 1;
            assignments[2*j] = ((z2 >> j) & 1u) + 1;
          }

          data(z1, z2) = eltC(contraction, assignments);
        }
      }

      return data;
    }

    ITensor partial_contraction(size_t _q1, size_t _q2, std::optional<std::vector<ITensor>> contraction_ops_opt=std::nullopt) const {
      std::vector<ITensor> contraction_ops;
      bool has_ops = contraction_ops_opt.has_value();
      if (has_ops) {
        contraction_ops = contraction_ops_opt.value();
      }
      size_t q1 = std::min(_q1, _q2);
      size_t q2 = std::max(_q1, _q2);
      
      ITensor contraction;
      if (q1 == 0) {
        if (has_ops) {
          contraction = apply_block_r(left_block * ops[0] * contraction_ops[0] * conj(prime(ops[0])), 0);
        } else {
          contraction = apply_block_r(left_block * ops[0] * conj(prime(ops[0])), 0);
        }
      } else {
        if (has_ops) {
          contraction = apply_block_r(ops[q1] * contraction_ops[0] * conj(prime(ops[q1])), q1);
        } else {
          contraction = apply_block_r(ops[q1] * conj(prime(ops[q1])), q1);
        }

        contraction = apply_block_l(contraction, q1);
      }

      size_t k = 1;
      for (size_t q = q1 + 1; q < q2; q++) {
        if (has_ops) {
          contraction *= ops[q] * contraction_ops[k] * conj(prime(ops[q]));
          contraction = apply_block_r(contraction, q);
          k++;
        } else {
          contraction *= ops[q] * conj(prime(ops[q]));
          contraction = apply_block_r(contraction, q);
        }
      }

      return contraction;
    }

    double expectation(const PauliString& p) const {
      if (p.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but MatrixProductOperator has {} external qubits.", p.num_qubits, num_qubits));
      }

      std::vector<ITensor> paulis(num_qubits);
      for (size_t i = 0; i < num_qubits; i++) {
        Index idx = external_idx(i);
        paulis[i] = pauli_matrix(p.to_pauli(i), idx, prime(idx));
      }

      auto contraction = partial_contraction(0, num_qubits, paulis);

      std::vector<int> _inds;
      double sign = p.get_r() ? -1.0 : 1.0;
      return sign*tensor_to_scalar(contraction).real();
    }

    // Check that MPO is bipartite, with the traced qubits on the left side 
    bool left_bipartite() const {
      for (size_t i = 0; i < num_qubits; i++) {
        if (!block_trivial(i)) {
          return false;
        }
      }
      return true;
    }


    std::pair<PauliString, double> sample_pauli(std::minstd_rand& rng) const {
      std::vector<Pauli> p(num_qubits);
      double P = 1.0;

      ITensor L = left_block / std::sqrt(tensor_to_scalar(left_block * left_block).real());

      for (size_t k = 0; k < num_qubits; k++) {
        std::vector<double> probs(4);
        std::vector<ITensor> tensors(4);

        auto Ak = ops[k];
        std::string label1 = fmt::format("n={},Left", k);
        std::string label2 = fmt::format("n={},Right", k);
        Index alpha_left = findInds(Ak, label1)[0];
        L.replaceInds(inds(L), {alpha_left, prime(alpha_left)});

        Index s = external_indices[k];

        for (size_t p = 0; p < 4; p++) {
          auto sigma = pauli_matrix(static_cast<Pauli>(p), s, prime(s));

          auto C = prime(Ak)*conj(Ak)*sigma*L;
          auto contraction = conj(C)*C / 2.0;
          
          std::vector<size_t> inds;
          double prob = std::abs(eltC(contraction, inds));
          probs[p] = prob;
          tensors[p] = C / std::sqrt(2.0 * prob);
        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        size_t a = dist(rng);

        p[k] = static_cast<Pauli>(a);
        P *= probs[a];
        L = apply_block_r(tensors[a], k);
      }

      double t = std::sqrt(P*std::pow(2.0, num_qubits));
      return std::make_pair(PauliString(p), t);
    }

    std::vector<PauliAmplitude> sample_paulis(size_t num_samples, std::minstd_rand& rng) const {
      if (!left_bipartite()) {
        throw std::runtime_error("Cannot sample_paulis for a non-left-bipartite MPO.");
      }
      std::vector<PauliAmplitude> samples(num_samples);

      for (size_t k = 0; k < num_samples; k++) {
        samples[k] = sample_pauli(rng);
      } 

      return samples;
    }
    
  private:
    static std::string block_to_string(const MPOBlock& block) {
      if (block) {
        std::stringstream s;
        s << block.value();
        return s.str();
      } else {
        return "None\n";
      }
    }
};

// ----------------------------------------------------------------------- //
// --------------- MatrixProductState implementation --------------------- //
// ----------------------------------------------------------------------- //

MatrixProductState::MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) : QuantumState(num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(num_qubits, bond_dimension, sv_threshold);
}

MatrixProductState::MatrixProductState(const MatrixProductState& other) : QuantumState(other.num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(*other.impl.get());
}

MatrixProductState::MatrixProductState(const Statevector& other, uint32_t bond_dimension, double sv_threshold) : MatrixProductState(other.num_qubits, bond_dimension, sv_threshold) {
  auto coefficients = vector_to_tensor(other.data, impl->external_indices);
  impl->reset_from_tensor(coefficients);
}

MatrixProductState::~MatrixProductState()=default;

MatrixProductState& MatrixProductState::operator=(const MatrixProductState& other) {
  if (this != &other) {
    impl = std::make_unique<MatrixProductStateImpl>(*other.impl);
  }
  return *this;
}

MatrixProductState MatrixProductState::ising_ground_state(size_t num_qubits, double h, size_t bond_dimension, double sv_threshold, size_t num_sweeps) {
  SiteSet sites = SpinHalf(num_qubits, {"ConserveQNs=",false});

  auto ampo = AutoMPO(sites);
  for(int j = 1; j < num_qubits; ++j) {
    ampo += -2.0, "Sx", j, "Sx", j + 1;
  }

  for(int j = 1; j <= num_qubits; ++j) {
    ampo += -h, "Sz", j;
  }
  auto H = toMPO(ampo);

  auto psi = randomMPS(sites, bond_dimension);
  auto sweeps = Sweeps(num_sweeps);
  sweeps.maxdim() = bond_dimension;
  sweeps.cutoff() = sv_threshold;
  sweeps.noise() = 1E-8;

  auto [energy, psi0] = dmrg(H, psi, sweeps, {"Silent=",true});
  psi0.normalize();

  auto impl = std::make_unique<MatrixProductStateImpl>(MatrixProductStateImpl::from_mps(psi0, bond_dimension, sv_threshold));
  impl->bond_dimension = bond_dimension;
  impl->sv_threshold = sv_threshold;

  MatrixProductState mps(num_qubits, bond_dimension, sv_threshold);
  mps.impl = std::move(impl);

  Eigen::Matrix4cd id;
  id.setIdentity();
  for (uint32_t i = 0; i < num_qubits-1; i++) {
    mps.evolve(id, {i, i+1});
  }

  return mps;
}

std::string MatrixProductState::to_string() const {
  return impl->to_string();
}

double MatrixProductState::entropy(const std::vector<uint32_t>& qubits, uint32_t index) {
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

	uint32_t q = sorted_qubits.back() + 1;

	return impl->entropy(q, index);
}

// TODO revist; see if can sample MPS with PDF ~<P>^4 (default is <P>^2)
double MatrixProductState::magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  auto contiguous = [](const std::vector<uint32_t>& v) {
    auto v_r = v;
    std::sort(v_r.begin(), v_r.end());

    for (size_t i = 0; i < v_r.size() - 1; i++) {
      if (v_r[i+1] != v_r[i] + 1) {
        return false;
      }
    }

    return true;
  };

  if (!contiguous(qubitsA) || !contiguous(qubitsB)) {
    throw std::runtime_error(fmt::format("qubitsA = {}, qubitsB = {} not contiguous. Can't compute MPS.magic_mutual_information.", qubitsA, qubitsB));
  }

  std::vector<bool> mask(num_qubits, false);
  for (const uint32_t q : qubitsA) {
    mask[q] = true;
  }

  for (const uint32_t q : qubitsB) {
    mask[q] = true;
  }

  for (size_t i = 0; i < num_qubits; i++) {
    if (!mask[i]) {
      throw std::runtime_error(fmt::format("qubitsA = {}, qubitsB = {} are not a bipartition of system with {} qubits. Can't compute MPS.magic_mutual_information.", qubitsA, qubitsB, num_qubits));
    }
  }

  std::vector<uint32_t> qubitsB_ = qubitsB;
  for (size_t i = 0; i < qubitsB_.size(); i++) {
    qubitsB_[i] = num_qubits - qubitsB_[i] - 1;
  }

  std::sort(qubitsB_.begin(), qubitsB_.end());

  auto stateA = partial_trace_mpo(qubitsA);
  auto stateB = partial_trace_mpo(qubitsB);

  auto samplesAB = sample_paulis(num_samples);
  // Why does this work for MPO states? Should only work if left-bipartite?
  auto samplesA = stateA.sample_paulis(num_samples);
  auto samplesB = stateB.sample_paulis(num_samples);

  double MA = stateA.stabilizer_renyi_entropy(2, samplesA);
  double MB = stateB.stabilizer_renyi_entropy(2, samplesB);
  double MAB = stabilizer_renyi_entropy(2, samplesAB);

  return MAB - MA - MB;
}

std::vector<double> MatrixProductState::pauli_expectation_left_sweep(const PauliString& P, uint32_t q1, uint32_t q2) const {
  return QuantumState::pauli_expectation_left_sweep(P, q1, q2);
  return impl->pauli_expectation_left_sweep(P, q1, q2);
}

std::vector<double> MatrixProductState::pauli_expectation_right_sweep(const PauliString& P, uint32_t q1, uint32_t q2) const {
  return QuantumState::pauli_expectation_right_sweep(P, q1, q2);
  return impl->pauli_expectation_right_sweep(P, q1, q2);
}

std::vector<PauliAmplitude> MatrixProductState::sample_paulis(size_t num_samples) {
  return impl->sample_paulis(num_samples, rng);
}

double MatrixProductState::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

std::complex<double> MatrixProductState::expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const {
  return impl->expectation(m, sites);
}

void MatrixProductState::print_mps(bool print_data=false) const {
  impl->print_mps(print_data);
}

std::shared_ptr<QuantumState> MatrixProductState::partial_trace(const std::vector<uint32_t>& qubits) const {
  return std::make_shared<MatrixProductOperator>(*this, qubits);
}

MatrixProductOperator MatrixProductState::partial_trace_mpo(const std::vector<uint32_t>& qubits) const {
  return MatrixProductOperator(*this, qubits);
}

std::complex<double> MatrixProductState::coefficients(uint32_t z) const {
  return impl->coefficients(z);
}

Eigen::VectorXcd MatrixProductState::coefficients(const std::vector<uint32_t>& indices) const {
  return impl->coefficients(indices);
}

Eigen::VectorXcd MatrixProductState::coefficients() const {
  return impl->coefficients();
}

double MatrixProductState::trace() const {
  return impl->trace();
}

size_t MatrixProductState::bond_dimension(size_t i) const {
  return impl->bond_dimension_at_site(i);
}

void MatrixProductState::reverse() {
  impl->reverse();
}

std::complex<double> MatrixProductState::inner(const MatrixProductState& other) const {
  return impl->inner(*other.impl.get());
}

void MatrixProductState::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
  impl->evolve(gate, qubit);
}

void MatrixProductState::evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) {
  impl->evolve(gate, qubits);
}

void MatrixProductState::id(uint32_t q1, uint32_t q2) {
  impl->evolve(Eigen::Matrix4cd::Identity(), {q1, q2});
}

bool MatrixProductState::mzr(uint32_t q) {
  return impl->measure(q, randf());
}

std::vector<bool> MatrixProductState::measure(const std::vector<MeasurementData>& measurements) {
  std::vector<double> random_vals(measurements.size());
  for (size_t i = 0; i < measurements.size(); i++) {
    random_vals[i] = randf();
  }

  return impl->measure(measurements, random_vals);
}

bool MatrixProductState::measure(const PauliString& p, const std::vector<uint32_t>& qubits) {
  return impl->measure(p, qubits, randf());
}

std::vector<bool> MatrixProductState::weak_measure(const std::vector<WeakMeasurementData>& measurements) {
  std::vector<double> random_vals(measurements.size());
  for (size_t i = 0; i < measurements.size(); i++) {
    random_vals[i] = randf();
  }

  return impl->weak_measure(measurements, random_vals);
}

bool MatrixProductState::weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta) {
  return impl->weak_measure(p, qubits, beta, randf());
}

void MatrixProductState::show_problem_sites() const {
  impl->show_problem_sites();
}

std::vector<size_t> MatrixProductState::orthogonal_sites() const {
  return impl->orthogonal_sites_l();
}

void MatrixProductState::id_debug(uint32_t i, uint32_t j) {
  impl->id_debug(i, j);
}

bool MatrixProductState::debug_tests() {
  bool b1 = impl->check_orthonormality();
  bool b2 = impl->state_valid();

  if (!b1) {
    std::cout << "MPS is not orthonormal.\n";
  }

  if (!b2) {
    std::cout << "MPS has invalid singular values.\n";
  } 

  bool b = b1 && b2;
  if (!b) {
    impl->show_problem_sites();
  }

  return b;
}

std::vector<double> MatrixProductState::bipartite_magic_mutual_information(size_t num_samples) { 
  auto pauli_samples = sample_paulis(num_samples);

  size_t N = num_qubits/2 - 1;
  std::vector<std::vector<double>> samplesA(N);
  std::vector<std::vector<double>> samplesB(N);
  std::vector<std::vector<double>> samplesAB(N);

  for (size_t i = 0; i < num_samples; i++) {
    auto const [P, t] = pauli_samples[i];

    std::vector<double> tA = pauli_expectation_left_sweep(P, 0, num_qubits/2);
    std::vector<double> tB = pauli_expectation_right_sweep(P, num_qubits/2, 0);
    std::reverse(tB.begin(), tB.end());

    for (size_t j = 0; j < N; j++) {
      samplesA[j].push_back(tA[j]);
      samplesB[j].push_back(tB[j]);
      samplesAB[j].push_back(t);
    }
  }

  std::vector<double> magic(N);
  for (size_t j = 0; j < N; j++) {
    magic[j] = QuantumState::calculate_magic_mutual_information_from_chi_samples({samplesA[j], samplesB[j], samplesAB[j]});
  }

  return magic;
}

// ----------------------------------------------------------------------- //
// ------------ MatrixProductOperator implementation --------------------- //
// ----------------------------------------------------------------------- //

MatrixProductOperator::MatrixProductOperator(const MatrixProductState& mps, const std::vector<uint32_t>& traced_qubits) : QuantumState(mps.num_qubits - traced_qubits.size()) {
  impl = std::make_unique<MatrixProductOperatorImpl>(*mps.impl.get(), traced_qubits);
}

MatrixProductOperator::MatrixProductOperator(const MatrixProductOperator& mpo, const std::vector<uint32_t>& traced_qubits) : QuantumState(mpo.num_qubits - traced_qubits.size()) {
  impl = std::make_unique<MatrixProductOperatorImpl>(*mpo.impl.get(), traced_qubits);
}

MatrixProductOperator::MatrixProductOperator(const MatrixProductOperator& other) : QuantumState(other.num_qubits) {
  impl = std::make_unique<MatrixProductOperatorImpl>(*other.impl.get());
}

MatrixProductOperator::~MatrixProductOperator()=default;

MatrixProductOperator& MatrixProductOperator::operator=(const MatrixProductOperator& other) {
  if (this != &other) {
    impl = std::make_unique<MatrixProductOperatorImpl>(*other.impl);
  }
  return *this;
}

void MatrixProductOperator::print_mps() const {
  impl->print_mps();
}

Eigen::MatrixXcd MatrixProductOperator::coefficients() const {
  return impl->coefficients();
}

double MatrixProductOperator::trace() const {
  return impl->trace();
}

std::vector<PauliAmplitude> MatrixProductOperator::sample_paulis(size_t num_samples) {
  return impl->sample_paulis(num_samples, rng);
}

double MatrixProductOperator::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

std::shared_ptr<QuantumState> MatrixProductOperator::partial_trace(const std::vector<uint32_t>& qubits) const {
  return std::make_shared<MatrixProductOperator>(*this, qubits);
}

MatrixProductOperator MatrixProductOperator::partial_trace_mpo(const std::vector<uint32_t>& qubits) const {
  return MatrixProductOperator(*this, qubits);
}

