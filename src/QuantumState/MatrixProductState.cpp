#include "QuantumStates.h"
#include <itensor/all.h>
#include <sstream>

#include <fmt/ranges.h>
#include <random>

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

static bool is_identity(ITensor& I) {
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

    if (std::abs(eltC(I, idx_assignments) - 1.0) > 1e-4) {
      return false;
    }
  }

  return true;
}

static ITensor pauli_matrix(size_t i, Index i1, Index i2) {
  static Eigen::Matrix2cd I = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, 1.0).finished();
  static Eigen::Matrix2cd X = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished();
  static Eigen::Matrix2cd Y = (Eigen::Matrix2cd() << 0.0, std::complex<double>(0.0, -1.0), std::complex<double>(0.0, 1.0), 0.0).finished();
  static Eigen::Matrix2cd Z = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, -1.0).finished();

  if (i == 0) {
    return matrix_to_tensor(I, i1, i2);
  } else if (i == 1) {
    return matrix_to_tensor(X, i1, i2);
  } else if (i == 2) {
    return matrix_to_tensor(Y, i1, i2);
  } else if (i == 3) {
    return matrix_to_tensor(Z, i1, i2);
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

    static Eigen::Matrix2cd zero_projector() {
      return (Eigen::Matrix2cd() << 1, 0, 0, 0).finished();
    }

    static Eigen::Matrix2cd one_projector() {
      return (Eigen::Matrix2cd() << 0, 0, 0, 1).finished();
    }

    static Eigen::Matrix4cd SWAP() {
      return (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished();
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
        internal_indices.push_back(Index(1, fmt::format("Internal,Left,a{}", i)));
        internal_indices.push_back(Index(1, fmt::format("Internal,Right,a{}", i)));
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        external_indices.push_back(Index(2, fmt::format("External,i{}", i)));
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

    MatrixProductStateImpl(const MPS& mps_) : MatrixProductStateImpl(mps_.length(), 32u, 1e-8) {
      MPS mps(mps_);
      mps.position(0);

      ITensor U;
      ITensor V = mps(1);
      ITensor S;
      for (size_t i = 1; i < num_qubits; i++) {

        std::vector<Index> u_inds{siteIndex(mps, i)};
        std::vector<Index> v_inds{siteIndex(mps, i+1)};
        if (i != 1) {
          u_inds.push_back(internal_idx(i - 2, InternalDir::Right));
        }

        if (i != mps.length() - 1) {
          v_inds.push_back(linkIndex(mps, i + 1));
        }


        auto M = V*mps(i + 1);

        std::tie(U, S, V) = svd(M, u_inds, v_inds,
            {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
             "LeftTags=",fmt::format("a{},Internal,Left",i-1),
             "RightTags=",fmt::format("a{},Internal,Right",i-1)});

        auto ext = siteIndex(mps, i);
        U.replaceTags(ext.tags(), external_idx(i - 1).tags());
        if (i != 1) {
          U.replaceTags(linkIndex(mps, i-1).tags(), tags(internal_idx(i-1, InternalDir::Left)));
        }

        if (i != num_qubits-1) {
          U.replaceTags(linkIndex(mps, i).tags(), tags(internal_idx(i, InternalDir::Right)));
        }

        tensors[i-1] = U;
        singular_values[i-1] = S;

        external_indices[i-1] = findInds(U, "External")[0];
        internal_indices[2*(i-1)] = findInds(S, "Internal,Left")[0];
        internal_indices[2*(i-1)+1] = findInds(S, "Internal,Right")[0];
      }

      V.replaceTags(siteIndex(mps, num_qubits).tags(), external_idx(num_qubits - 1).tags());
      external_indices[num_qubits - 1] = findInds(V, "External")[0];
      tensors[num_qubits - 1] = V;
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

    ITensor A(size_t i, bool right_normalized=true) const {
      auto Ai = tensors[i];
      if (right_normalized) {
        if (i != 0) {
          Ai *= singular_values[i - 1];
        }
      } else {
        if (i != num_qubits - 1) {
          Ai *= singular_values[i];
        }
      }

      Index left;
      Index right;

      Index _left = (i > 0) ? internal_idx(i-1, InternalDir::Right) : Index();
      Index _right = (i < num_qubits - 1) ? internal_idx(i, InternalDir::Right) : Index();

      if (i == 0) {
        left = Index(1, "alpha0,Internal");
        right = Index(dim(_right), "alpha1,Internal");
        Ai.replaceInds({_right}, {right});

        ITensor one(left);
        one.set(left=1, 1.0);
        Ai *= one;
      } else if (i == num_qubits - 1) {
        left = Index(dim(_left), fmt::format("alpha{},Internal", i));
        right = Index(1, fmt::format("alpha{},Internal",i+1));
        Ai.replaceInds({_left}, {left});

        ITensor one(right);
        one.set(right=1, 1.0);
        Ai *= one;
      } else {
        left = Index(dim(_left), fmt::format("alpha{},Internal", i));
        right = Index(dim(_right), fmt::format("alpha{},Internal",i+1));
        Ai.replaceInds({_left, _right}, {left, right});
      }

      return Ai;
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

        auto Ak = A(k, false);
        std::string label1 = fmt::format("alpha{}", k);
        std::string label2 = fmt::format("alpha{}", k+1);
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

    std::vector<PauliAmplitude> stabilizer_renyi_entropy_samples(size_t num_samples) {
      std::vector<PauliAmplitude> samples(num_samples);

      for (size_t k = 0; k < num_samples; k++) {
        samples[k] = sample_pauli();
      } 

      return samples;
    }

    std::complex<double> expectation(const PauliString& p) const {
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
      double sign = p.r() ? -1.0 : 1.0;
      return sign*eltC(contraction, _inds);
    }

    ITensor coefficient_tensor() const {
      ITensor C = tensors[0];

      for (uint32_t i = 0; i < num_qubits-1; i++) {
        C *= singular_values[i]*tensors[i+1];
      }

      //ITensor C = A(0, false);
      //for (uint32_t i = 1; i < num_qubits; i++) {
      //  auto Ai = A(i, false);
      //  auto s = fmt::format("alpha{}", i);
      //  Ai.replaceInds(findInds(Ai, s), findInds(C, s));
      //  C *= Ai;
      //}

      //C *= delta(findInds(C, "Internal"));

  

      return C;
    }

    void print_mps() const {
      print(tensors[0]);
      for (uint32_t i = 0; i < num_qubits - 1; i++) {
        print(singular_values[i]);
        print(tensors[i+1]);
      }
    }

    void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
      auto i = external_indices[qubit];
      auto ip = prime(i);
      ITensor tensor = matrix_to_tensor(gate, i, ip);
      tensors[qubit] = noPrime(tensors[qubit]*tensor);
    }

    void swap(uint32_t q1, uint32_t q2) {
      evolve(MatrixProductStateImpl::SWAP(), {q1, q2});
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
        evolve(gate, qbits);

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
          "LeftTags=",fmt::format("Internal,Left,a{}", q1),
          "RightTags=",fmt::format("Internal,Right,a{}", q1)});

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

      return real(sumelsC(qtensor * dag(qtensor)));
    }

    bool valid_state() const {
      for (size_t i = 0; i < num_qubits - 1; i++) {
        size_t d = dim(inds(singular_values[i])[0]);
        for (size_t j = 1; j <= d; j++) {
          double v = elt(singular_values[i], j, j);
          if (v > 1.0 + 1e-5 || std::isnan(v)) {
            PrintData(singular_values[i]);
            return false;
          }
        }
      }

      return true;
    }

    bool measure(uint32_t q, double r) {
      double prob_zero = measure_probability(q, 0);
      bool outcome = r >= prob_zero;

      Eigen::Matrix2cd proj = outcome ? 
        MatrixProductStateImpl::one_projector()/std::sqrt(1.0 - prob_zero) :
        MatrixProductStateImpl::zero_projector()/std::sqrt(prob_zero);

      evolve(proj, q);

      Eigen::Matrix4cd id;
      id.setIdentity();

      for (uint32_t i = q; i < num_qubits - 1; i++) {
        if (dim(internal_idx(i, InternalDir::Right)) == 1) {
          break;
        }

        evolve(id, {i, i+1});
      }

      // Propagate left
      for (uint32_t i = q; i > 1; i--) {
        if (dim(internal_idx(i-1, InternalDir::Left)) == 1) {
          break;
        }

        evolve(id, {i-1, i});
      }

      return outcome;
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
        return delta(i_r, prime(i_r));
      }

      auto Ak = A[0];
      auto a0 = findIndex(Ak, "alpha0");
      auto C = Ak*conj(prime(Ak, "Internal"))*delta(a0, prime(a0));
      for (size_t k = 1; k < q; k++) {
        Ak = A[k];
        C *= Ak*conj(prime(Ak, "Internal"));
      }

      Index i = noPrime(findInds(C, fmt::format("alpha{}", q))[0]);

      C.replaceInds({i, prime(i)}, {i_r, prime(i_r)});
      return C;
    }

    static MPOBlock get_block_right(const std::vector<ITensor>& A, size_t q, const Index& i_l) {
      size_t L = A.size();
      if (q == L) {
        return delta(i_l, prime(i_l));
      }

      auto Ak = A[L - 1];
      auto aL = findIndex(Ak, fmt::format("alpha{}", L));
      auto C = Ak*conj(prime(Ak, "Internal"))*delta(aL, prime(aL));

      for (size_t k = L - 2; k > q; k--) {
        Ak = A[k];
        C *= Ak*conj(prime(Ak, "Internal"));
      }

      Index i = noPrime(findInds(C, fmt::format("alpha{}", q+1))[0]);

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

      Index i1 = noPrime(findInds(C, fmt::format("alpha{}", q1+1))[0]);
      Index i2 = noPrime(findInds(C, fmt::format("alpha{}", q2))[0]);

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

    MatrixProductOperatorImpl(const MatrixProductStateImpl& mps, const std::vector<uint32_t>& traced_qubits)
      : num_qubits(mps.num_qubits - traced_qubits.size()), bond_dimension(mps.bond_dimension), sv_threshold(mps.sv_threshold) {

      std::vector<bool> mask(mps.num_qubits, false);
      for (auto const q : traced_qubits) {
        mask[q] = true;
      }

      // Generate all tensors
      std::vector<ITensor> A;
      for (size_t i = 0; i < mps.num_qubits; i++) {
        A.push_back(mps.A(i, false));
      }
      
      // Align indices
      for (size_t i = 1; i < mps.num_qubits; i++) {
        std::string s = fmt::format("alpha{}", i);
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
          Index external_idx_(dim(external_idx), fmt::format("i{},External",k));

          Index internal_idx1 = findIndex(Ai, fmt::format("alpha{}", i));
          Index internal_idx2 = findIndex(Ai, fmt::format("alpha{}", i+1));
          Index internal_idx1_ = Index(dim(internal_idx1), fmt::format("a{},Internal,Left", k));
          Index internal_idx2_ = Index(dim(internal_idx2), fmt::format("a{},Internal,Right", k));

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
    
    std::complex<double> expectation(const PauliString& p) const {
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

      ITensor contraction = left_block;

      for (size_t i = 0; i < num_qubits; i++) {
        contraction = apply_block(contraction*ops[i]*paulis[i]*conj(prime(ops[i])), i);
      }

      std::vector<int> _inds;
      double sign = p.r() ? -1.0 : 1.0;
      return sign*eltC(contraction, _inds);
    }

    ITensor apply_block(const ITensor& tensor, size_t i) const {
      if (blocks[i]) {
        return tensor*blocks[i].value();
      } else {
        Index i1 = internal_idx(i, InternalDir::Right);
        Index i2 = internal_idx(i+1, InternalDir::Left);
        return replaceInds(tensor, {i1, prime(i1)}, {i2, prime(i2)});
      }
    }

    Eigen::MatrixXcd coefficients() const {
      if (num_qubits > 31) {
        throw std::runtime_error("Cannot generate coefficients for n > 31 qubits.");
      }

      ITensor contraction = left_block;

      for (size_t i = 0; i < num_qubits; i++) {
        contraction *= ops[i]*prime(conj(ops[i]));
        contraction = apply_block(contraction, i);
      }

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

MatrixProductState::MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) : QuantumState(num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(num_qubits, bond_dimension, sv_threshold);
  impl->seed(rand());
}

MatrixProductState::MatrixProductState(const MatrixProductState& mps) : QuantumState(mps.num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(*mps.impl.get());
  impl->seed(rand());
}

MatrixProductState::~MatrixProductState()=default;

MatrixProductState MatrixProductState::ising_ground_state(size_t num_qubits, double h, size_t bond_dimension, double sv_threshold, size_t num_sweeps) {
  SiteSet sites = SpinHalf(num_qubits, {"ConserveQNs=",false});

  auto ampo = AutoMPO(sites);
  for(int j = 1; j < num_qubits; ++j) {
    ampo += -2.0, "Sz", j, "Sz", j + 1;
  }

  for(int j = 1; j <= num_qubits; ++j) {
    ampo += -h, "Sx", j;
  }
  auto H = toMPO(ampo);

  auto psi = randomMPS(sites);
  auto sweeps = Sweeps(num_sweeps);
  sweeps.maxdim() = bond_dimension;
  sweeps.cutoff() = sv_threshold;
  sweeps.noise() = 1E-8;

  auto [energy, psi0] = dmrg(H, psi, sweeps, {"Silent=",true});

  auto impl = std::make_unique<MatrixProductStateImpl>(psi0);
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

void MatrixProductState::seed(int i) {
  QuantumState::seed(i);
  impl->seed(rand());
}

std::string MatrixProductState::to_string() const {
	Statevector state(*this);
	return state.to_string();
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

std::vector<PauliAmplitude> MatrixProductState::stabilizer_renyi_entropy_samples(size_t num_samples) {
  return impl->stabilizer_renyi_entropy_samples(num_samples);
}


std::vector<double> MatrixProductState::magic_mutual_information(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples) {
  std::vector<uint32_t> qubitsAB;

  qubitsAB.insert(qubitsAB.end(), qubitsA.begin(), qubitsA.end());
  qubitsAB.insert(qubitsAB.end(), qubitsB.begin(), qubitsB.end());
  std::vector<uint32_t> qubitsA_(qubitsA.size());
  std::vector<uint32_t> qubitsB_(qubitsB.size());
  std::iota(qubitsA_.begin(), qubitsA_.end(), 0);
  std::iota(qubitsB_.begin(), qubitsB_.end(), 0);

  MatrixProductOperator mpsAB = partial_trace(qubitsAB);
  auto samples = mpsAB.stabilizer_renyi_entropy_samples(num_samples);
  std::vector<double> magic_samples;
  for (const auto& [P, p] : samples) {
    PauliString PA = P.substring(qubitsA_, false);
    PauliString PB = P.substring(qubitsB_, false);

    double tAB = std::abs(mpsAB.expectation(P));
    double tA = std::abs(mpsAB.expectation(PA));
    double tB = std::abs(mpsAB.expectation(PB));

    magic_samples.push_back(tAB/(tA*tB));
  }

  return magic_samples;
}

std::complex<double> MatrixProductState::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

void MatrixProductState::print_mps() const {
  impl->print_mps();
}

MatrixProductOperator MatrixProductState::partial_trace(const std::vector<uint32_t>& qubits) const {
  return MatrixProductOperator(*this, qubits);
}

std::complex<double> MatrixProductState::coefficients(uint32_t z) const {
	auto C = impl->coefficient_tensor();

	std::vector<int> assignments(num_qubits);
	for (uint32_t j = 0; j < num_qubits; j++) {
		assignments[j] = ((z >> j) & 1u) + 1;
	}

	return eltC(C, assignments);
}

Eigen::VectorXcd MatrixProductState::coefficients(const std::vector<uint32_t>& indices) const {
  if (num_qubits > 31) {
    throw std::runtime_error("Cannot generate coefficients for n > 31 qubits.");
  }

  auto C = impl->coefficient_tensor();

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

Eigen::VectorXcd MatrixProductState::coefficients() const {
	std::vector<uint32_t> indices(1u << num_qubits);
	std::iota(indices.begin(), indices.end(), 0);
	
	return coefficients(indices);
}

void MatrixProductState::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
  impl->evolve(gate, qubit);
}


void MatrixProductState::evolve(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) {
  impl->evolve(gate, qubits);
}

double MatrixProductState::measure_probability(uint32_t q, bool outcome) const {
  return impl->measure_probability(q, outcome);
}

bool MatrixProductState::measure(uint32_t q) {
  double r = randf();
  return impl->measure(q, r);
}

MatrixProductOperator::MatrixProductOperator(const MatrixProductState& mps, const std::vector<uint32_t>& traced_qubits) : QuantumState(mps.num_qubits - traced_qubits.size()) {
  impl = std::make_unique<MatrixProductOperatorImpl>(*mps.impl.get(), traced_qubits);
}

MatrixProductOperator::~MatrixProductOperator()=default;

void MatrixProductOperator::print_mps() const {
  impl->print_mps();
}

Eigen::MatrixXcd MatrixProductOperator::coefficients() const {
  return impl->coefficients();
}

std::complex<double> MatrixProductOperator::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

