#include "QuantumStates.h"

#include <sstream>
#include <random>

#include <fmt/ranges.h>
#include <itensor/all.h>
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

static bool is_identity(const ITensor& I, double tol=1e-4) {
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

static ITensor pauli_matrix(size_t i, Index i1, Index i2) {
  if (i == 0) {
    return matrix_to_tensor(quantumstate_utils::I::value, i1, i2);
  } else if (i == 1) {
    return matrix_to_tensor(quantumstate_utils::X::value, i1, i2);
  } else if (i == 2) {
    return matrix_to_tensor(quantumstate_utils::Y::value, i1, i2);
  } else if (i == 3) {
    return matrix_to_tensor(quantumstate_utils::Z::value, i1, i2);
  }

  throw std::runtime_error(fmt::format("Invalid Pauli index {}.", i));
}

class MatrixProductStateImpl {
  friend class MatrixProductState;

  private:
    std::mt19937 rng;

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

    void seed(int i) {
      rng.seed(i);
    }

    MatrixProductStateImpl(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) 
    : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold) {
      std::random_device random_device;
      seed(random_device());

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


    double entropy(uint32_t q) {
      if (q < 0 || q > num_qubits) {
        throw std::invalid_argument("Invalid qubit passed to MatrixProductState.entropy; must have 0 <= q <= num_qubits.");
      }

      if (q == 0 || q == num_qubits) {
        return 0.0;
      }

      auto sv = singular_values[q-1];
      int d = dim(inds(sv)[0]);

      double s = 0.0;
      for (int i = 1; i <= d; i++) {
        double v = std::pow(elt(sv, i, i), 2);
        if (v >= 1e-6) {
          s -= v * std::log(v);
        }
      }

      return s;
    }

    ITensor A_l(size_t i) const {
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

    ITensor A_r(size_t i) const {
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

    bool check_orthonormality() const {
      for (size_t i = 0; i < num_qubits; i++) {
        auto A = A_l(i);

        auto I = orthogonality_tensor_l(i);
        if (!is_identity(I, 1e-1)) {
          return false;
        }
      }

      // TODO check right normalization
      //for (size_t i = 0; i < num_qubits; i++) {
      //  auto A = A_r(i);

      //  Index idx = findIndex(A, fmt::format("m={}", i+1));
      //  auto I = A * conj(prime(A, idx));
      //  if (!is_identity(I)) {
      //    std::cout << fmt::format("At site {}, (R)\n", i);
      //    PrintData(I);
      //    return false;
      //  }
      //}

      return true;
    }

    std::pair<PauliString, double> sample_pauli() {
      std::vector<Pauli> p(num_qubits);
      double P = 1.0;

      Index i(1, "i");
      Index j(1, "j");

      ITensor L(i, j);
      L.set(i=1, j=1, 1.0);

      for (size_t k = 0; k < num_qubits; k++) {
        std::vector<double> probs(4);
        std::vector<ITensor> tensors(4);

        auto Ak = A_l(k);
        std::string label1 = fmt::format("m={}", k);
        std::string label2 = fmt::format("m={}", k+1);
        Index alpha_left = findInds(Ak, label1)[0];
        L.replaceInds(inds(L), {alpha_left, prime(alpha_left)});

        Index s = external_indices[k];

        for (size_t p = 0; p < 4; p++) {
          auto sigma = pauli_matrix(p, s, prime(s));

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

    std::vector<PauliAmplitude> sample_paulis(size_t num_samples) {
      std::vector<PauliAmplitude> samples(num_samples);

      for (size_t k = 0; k < num_samples; k++) {
        samples[k] = sample_pauli();
      } 

      return samples;
    }

    double expectation(const PauliString& p) const {
      if (p.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but MatrixProductState has {} qubits.", p.num_qubits, num_qubits));
      }

      std::vector<ITensor> paulis(num_qubits);
      for (size_t i = 0; i < num_qubits; i++) {
        Index idx = external_indices[i];
        if (p.to_op(i) == "I") {
          paulis[i] = pauli_matrix(0, idx, prime(idx));
        } else if (p.to_op(i) == "X") {
          paulis[i] = pauli_matrix(1, idx, prime(idx));
        } else if (p.to_op(i) == "Y") {
          paulis[i] = pauli_matrix(2, idx, prime(idx));
        } else {
          paulis[i] = pauli_matrix(3, idx, prime(idx));
        }
      }

      ITensor C = tensors[0];
      ITensor contraction = C*paulis[0]*prime(conj(C));

      for (size_t i = 1; i < num_qubits; i++) {
        C = tensors[i]*singular_values[i-1];
        contraction *= C*paulis[i]*prime(conj(C));
      }

      std::vector<int> _inds;
      double sign = p.get_r() ? -1.0 : 1.0;
      return sign*eltC(contraction, _inds).real();
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
        if (sites_[i] != sites[i - 1] + 1) {
          throw std::runtime_error(fmt::format("Provided sites {} are not contiguous.", sites_));
        }
      }

      size_t q1 = sites_[0];
      size_t q2 = sites_[n - 1];

      ITensor m_tensor;
      if (n == 1) {
        Index i = external_idx(q1);
        m_tensor = matrix_to_tensor(m, prime(i), i);

        ITensor contraction = tensors[q1] * prime(conj(tensors[q1])) * m_tensor;

        if (q1 != 0) {
          ITensor C = toDense(singular_values[q1 - 1]);
          contraction = contraction * C * conj(prime(C));
          Index left = noPrime(findInds(contraction, fmt::format("n={},Left", q1-1))[0]);
          contraction *= delta(left, prime(left));
        }

        if (q1 != num_qubits - 1) {
          ITensor C = toDense(singular_values[q1]);
          contraction = contraction * C * conj(prime(C));
          Index right = noPrime(findInds(contraction, fmt::format("n={},Right", q1))[0]);
          contraction *= delta(right, prime(right));
        }

        return tensor_to_scalar(contraction);
      } else if (n == 2) {
        Index i = external_idx(q1);
        Index j = external_idx(q2);
        m_tensor = matrix_to_tensor(m, prime(i), prime(j), i, j);

        // More efficient local contraction
        ITensor A1 = A_r(q1);
        ITensor A2 = A_l(q2);

        Index left = internal_idx(q1, InternalDir::Left);
        Index right = internal_idx(q1, InternalDir::Right);
        Index left_ = findIndex(A1, fmt::format("m={}",q2));
        Index right_ = findIndex(A2, fmt::format("m={}",q2));


        // TOOD remove toDense?
        ITensor C = A1 * replaceInds(toDense(singular_values[q1]), {left, right}, {left_, right_});
        ITensor contraction = C * conj(prime(C)) * m_tensor * A2 * prime(conj(A2));
        
        left = findIndex(A1, fmt::format("m={}", q1));
        right = findIndex(A2, fmt::format("m={}", q2+1));
        contraction *= delta(left, prime(left));
        contraction *= delta(right, prime(right));

        auto c = tensor_to_scalar(contraction);
        return tensor_to_scalar(contraction);
      } else {
        throw std::runtime_error("Currectly only support 1- and 2- qubit expectations.");
      }
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

    std::string to_string() const {
      Statevector sv(coefficients());
      return sv.to_string();
    }

    std::vector<double> get_norms() const {
      std::vector<double> norms{norm(tensors[0])};
      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        norms.push_back(norm(singular_values[i]));
        norms.push_back(norm(tensors[i+1]));
      }
      return norms;
    }

    void print_norms() const {
      std::cout << fmt::format("tensors[0] = {}\n", norm(tensors[0]));

      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        std::cout << fmt::format("singular_values[{}] = {}\n", i, norm(singular_values[i]));
        std::cout << fmt::format("tensors[{}] = {}\n", i+1, norm(tensors[i+1]));
      }

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

    void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
      auto i = external_indices[qubit];
      auto ip = prime(i);
      ITensor tensor = matrix_to_tensor(gate, i, ip);
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

      if (qubits.size() != 2) {
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

      ITensor theta = noPrime(gate_tensor*tensors[q1]*singular_values[q1]*tensors[q2]);

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

    double mzr_prob(uint32_t q, bool outcome) const {
      int i = static_cast<int>(outcome) + 1;
      auto idx = external_indices[q];
      auto qtensor = tensor_slice(tensors[q], idx, i);

      if (q > 0) {
        qtensor *= singular_values[q-1];
      }

      if (q < num_qubits - 1) {
        qtensor *= singular_values[q];
      }

      return real(sumelsC(qtensor * dag(qtensor)));
    }

    bool valid_state() const {
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

    double trace(size_t i) const {
      ITensor A = A_l(i);

      Index idx = findIndex(A, fmt::format("m={}", i));
      auto I = A * conj(prime(A, idx));
      auto I_inds = inds(I);
      
      auto It = I * delta(I_inds[0], I_inds[1]);

      return tensor_to_scalar(It).real();
    }


    double trace() const {
      Index ext = external_idx(0);
      ITensor C = tensors[0];
      ITensor contraction = C * conj(prime(C)) * delta(ext, prime(ext));

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        ext = external_idx(i + 1);
        C = singular_values[i]*tensors[i+1];
        contraction = contraction * C * conj(prime(C)) * delta(ext, prime(ext));
      }

      return std::abs(tensor_to_scalar(contraction));
    }

    size_t bond_dimension_at_site(size_t i) const {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot check bond dimension of site {} for MPS with {} sites.", i, num_qubits));
      }

      return dim(inds(singular_values[i])[0]);
    }

    bool weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta, double r) {
      if (qubits.size() != p.num_qubits) {
        throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
      }

      if (qubits.size() == 1 && p.to_pauli(0) == Pauli::Y) {
        Eigen::Matrix2cd T = quantumstate_utils::H::value * quantumstate_utils::sqrtZ::value;
        evolve(T.adjoint(), qubits[0]);
        PauliString Z("+Z");
        Z.set_r(p.get_r());
        bool b = weak_measure(Z, qubits, beta, r);
        evolve(T, qubits[0]);
        return b;
      }

      auto pm = p.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << p.num_qubits, 1u << p.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;

      double prob_zero = std::abs(expectation(proj0, qubits));

      bool outcome = r >= prob_zero;

      Eigen::MatrixXcd t = pm;
      if (outcome) {
        t = -t;
      }

      Eigen::MatrixXcd proj = (beta*t).exp();
      Eigen::MatrixXcd P = proj.pow(2);
      std::complex<double> c = expectation(P, qubits);
      proj = proj / std::sqrt(std::abs(c));
      evolve(proj, qubits);

      uint32_t q = *std::ranges::min_element(qubits);
      normalize(q);
      
      return outcome;
    }

    ITensor orthogonality_tensor_l(uint32_t i) const {
      ITensor A = A_l(i);
      return A * conj(prime(A, fmt::format("m={}",i)));
    }

    ITensor orthogonality_tensor_r(uint32_t i) const {
        ITensor A = A_r(i);
        return A * conj(prime(A, fmt::format("m={}",i)));
    }

    void normalize(size_t q) {
      //std::cout << fmt::format("q = {}\n\n", q);
      //for (size_t j = 0; j < num_qubits; j++) {
      //  std::cout << fmt::format("j = {}\n", j);
      //  PrintData(orthogonality_tensor_l(j));
      //  if (j != num_qubits - 1) {
      //    PrintData(singular_values[j]);
      //  }
      //}

      Eigen::Matrix4cd id = Eigen::Matrix4cd::Identity();

      // TODO check dimension of singular_values and stop propagation
      for (uint32_t i = q; i < num_qubits - 1; i++) {
        //if (dim(internal_idx(i, InternalDir::Right)) == 1) {
        //  break;
        //}

        evolve(id, {i, i+1});
      }

      for (uint32_t i = q; i > 0; i--) {
        //if (dim(internal_idx(i-1, InternalDir::Left)) == 1) {
        //  break;
        //}

        evolve(id, {i-1, i});
      }

      //for (size_t j = 0; j < num_qubits; j++) {
      //  std::cout << fmt::format("j = {}\n", j);
      //  PrintData(orthogonality_tensor_l(j));
      //  if (j != num_qubits - 1) {
      //    PrintData(singular_values[j]);
      //  }
      //}
    }

    bool measure(const PauliString& p, const std::vector<uint32_t>& qubits, double r) {
      if (qubits.size() != p.num_qubits) {
        throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
      }


      if (qubits.size() == 1) {
        Pauli op = p.to_pauli(0);
        if (op == Pauli::X) {
          evolve(quantumstate_utils::H::value, qubits[0]);
          PauliString Z("+Z");
          Z.set_r(p.get_r());
          bool b = measure(Z, qubits, r);
          evolve(quantumstate_utils::H::value, qubits[0]);
          return b;
        } else if (op == Pauli::Y) {
          Eigen::Matrix2cd T = quantumstate_utils::H::value * quantumstate_utils::sqrtZ::value;
          evolve(T.adjoint(), qubits[0]);
          PauliString Z("+Z");
          Z.set_r(p.get_r());
          bool b = measure(Z, qubits, r);
          evolve(T, qubits[0]);
          return b;
        }
      }

      auto pm = p.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << p.num_qubits, 1u << p.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;
      Eigen::MatrixXcd proj1 = (id - pm)/2.0;

      double prob_zero = std::abs(expectation(proj0, qubits));

      bool outcome = r >= prob_zero;

      proj0 = proj0/std::sqrt(prob_zero);
      proj1 = proj1/std::sqrt(1.0 - prob_zero);

      Eigen::MatrixXcd proj = outcome ? proj1 : proj0;

      evolve(proj, qubits);

      uint32_t q = *std::ranges::min_element(qubits);
      normalize(q);

      return outcome;
    }

    // TODO don't use measure(Z)
    bool measure(uint32_t q, double r) {
      PauliString Z(1);
      Z.set_z(0, 1);

      return measure(Z, {q}, r);

      //double prob_zero = mzr_prob(q, 0);
      //bool outcome = r >= prob_zero;

      //Eigen::Matrix2cd proj = outcome ? 
      //  MatrixProductStateImpl::one_projector()/std::sqrt(1.0 - prob_zero) :
      //  MatrixProductStateImpl::zero_projector()/std::sqrt(prob_zero);

      //evolve(proj, q);

      //propogate_normalization_left(q);
      //propogate_normalization_right(q);

      //return outcome;
    }
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
      std::vector<ITensor> A;
      for (size_t i = 0; i < mps.num_qubits; i++) {
        A.push_back(mps.A_l(i));
      }
      
      // Align indices
      for (size_t i = 1; i < mps.num_qubits; i++) {
        std::string s = fmt::format("m={}", i);
        auto ai = findInds(A[i - 1], s);
        A[i].replaceInds(findInds(A[i], s), findInds(A[i - 1], s));
      }

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
      size_t num_physical_qubits = mpo.num_qubits - num_traced_qubits;

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
    
    double expectation(const PauliString& p) const {
      if (p.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but MatrixProductOperator has {} external qubits.", p.num_qubits, num_qubits));
      }

      std::vector<ITensor> paulis(num_qubits);
      for (size_t i = 0; i < num_qubits; i++) {
        Index idx = external_idx(i);
        if (p.to_op(i) == "I") {
          paulis[i] = pauli_matrix(0, idx, prime(idx));
        } else if (p.to_op(i) == "X") {
          paulis[i] = pauli_matrix(1, idx, prime(idx));
        } else if (p.to_op(i) == "Y") {
          paulis[i] = pauli_matrix(2, idx, prime(idx));
        } else {
          paulis[i] = pauli_matrix(3, idx, prime(idx));
        }
      }

      auto contraction = partial_contraction(0, num_qubits, paulis);

      std::vector<int> _inds;
      double sign = p.get_r() ? -1.0 : 1.0;
      return sign*eltC(contraction, _inds).real();
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

    ITensor apply_block_r(const ITensor& tensor, size_t i) const {
      if (blocks[i]) {
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
      } else if (blocks[i-1]) {
        return tensor*blocks[i-1].value();
      } else {
        Index i1 = internal_idx(i, InternalDir::Left);
        Index i2 = internal_idx(i-1, InternalDir::Right);

        return replaceInds(tensor, {i1, prime(i1)}, {i2, prime(i2)});
      }

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

          auto c = eltC(contraction, assignments);

          data(z1, z2) = eltC(contraction, assignments);
        }
      }

      return data;
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
  impl->seed(rand());
}

MatrixProductState::MatrixProductState(const MatrixProductState& other) : QuantumState(other.num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(*other.impl.get());
  impl->seed(rand());
}

MatrixProductState::~MatrixProductState()=default;

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

void MatrixProductState::seed(int i, int j) {
  QuantumState::seed(i);
  impl->seed(j);
}

void MatrixProductState::seed(int i) {
  seed(i, rand());
}

std::string MatrixProductState::to_string() const {
  return impl->to_string();
}

double MatrixProductState::entropy(const std::vector<uint32_t>& qubits, uint32_t index) {
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

	uint32_t q = sorted_qubits.back() + 1;

	return impl->entropy(q);
}

magic_t MatrixProductState::magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt) {
  return magic_mutual_information_montecarlo_impl<MatrixProductState>(*this, qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
}

magic_t MatrixProductState::magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) {
  return magic_mutual_information_exhaustive_impl<MatrixProductState>(*this, qubitsA, qubitsB);
}

magic_t MatrixProductState::magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  return magic_mutual_information_exact_impl<MatrixProductState>(*this, qubitsA, qubitsB, num_samples);
}

std::vector<PauliAmplitude> MatrixProductState::sample_paulis(size_t num_samples) {
  return impl->sample_paulis(num_samples);
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

MatrixProductOperator MatrixProductState::partial_trace(const std::vector<uint32_t>& qubits) const {
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

void MatrixProductState::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
  impl->evolve(gate, qubit);
}

void MatrixProductState::evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) {
  impl->evolve(gate, qubits);
}

double MatrixProductState::mzr_prob(uint32_t q, bool outcome) const {
  return impl->mzr_prob(q, outcome);
}

bool MatrixProductState::mzr(uint32_t q) {
  return impl->measure(q, randf());
}

bool MatrixProductState::measure(const PauliString& p, const std::vector<uint32_t>& qubits) {
  return impl->measure(p, qubits, randf());
}

bool MatrixProductState::weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta) {
  return impl->weak_measure(p, qubits, beta, randf());
}

bool MatrixProductState::debug_tests() {
  bool b1 = impl->check_orthonormality();
  bool b2 = impl->valid_state();

  if (!b1) {
    std::cout << "Not orthonormal.\n";
  }

  if (!b2) {
    std::cout << "not valid.\n";
  } 

  return impl->check_orthonormality() && impl->valid_state();
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

void MatrixProductOperator::print_mps() const {
  impl->print_mps();
}

Eigen::MatrixXcd MatrixProductOperator::coefficients() const {
  return impl->coefficients();
}

double MatrixProductOperator::trace() const {
  return impl->trace();
}

double MatrixProductOperator::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

magic_t MatrixProductOperator::magic_mutual_information_montecarlo(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt) {
  return magic_mutual_information_montecarlo_impl<MatrixProductOperator>(*this, qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
}

magic_t MatrixProductOperator::magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) {
  return magic_mutual_information_exhaustive_impl<MatrixProductOperator>(*this, qubitsA, qubitsB);
}

magic_t MatrixProductOperator::magic_mutual_information_exact(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  return magic_mutual_information_exact_impl<MatrixProductOperator>(*this, qubitsA, qubitsB, num_samples);
}

MatrixProductOperator MatrixProductOperator::partial_trace(const std::vector<uint32_t>& qubits) const {
  return MatrixProductOperator(*this, qubits);
}

