#include "QuantumStates.h"
#include <itensor/all.h>

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

class MatrixProductStateImpl {
  friend class MatrixProductState;

  private:
    std::mt19937 rng;
    uint32_t num_qubits;
    uint32_t bond_dimension;
    double sv_threshold;

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

      //std::cout << fmt::format("s = {}\n", s);

      return s;
    }

    DensityMatrix partial_trace(const std::vector<uint32_t>& qubits) const {
      ITensor C = tensors[0];
      ITensor contraction = C*prime(conj(C));

      for (size_t i = 1; i < num_qubits; i++) {
        C = tensors[i]*singular_values[i-1];
        contraction *= C*prime(conj(C));
      }

      for (auto const q : qubits) {
        contraction *= delta(external_indices[q], prime(external_indices[q]));
      }

      size_t remaining_qubits = num_qubits - qubits.size();
      size_t s = 1u << remaining_qubits;
      Eigen::MatrixXcd data = Eigen::MatrixXcd::Zero(s, s);


      for (size_t z1 = 0; z1 < s; z1++) {
        for (size_t z2 = 0; z2 < s; z2++) {
          std::vector<int> assignments(2*remaining_qubits);
          for (size_t j = 0; j < remaining_qubits; j++) {
            assignments[j] = ((z1 >> j) & 1u) + 1;
            assignments[j + remaining_qubits] = ((z2 >> j) & 1u) + 1;
          }

          data(z1, z2) = eltC(contraction, assignments);
        }
      }

      return DensityMatrix(data);
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

    ITensor pauli_matrix(size_t i, Index i1, Index i2) const {
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
        throw std::runtime_error("Number of qubits in PauliString does not match number of qubits in MPS.");
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
  MatrixProductState mps(num_qubits, bond_dimension, 1.0);
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

  mps.impl = std::make_unique<MatrixProductStateImpl>(psi0);
  mps.impl->bond_dimension = bond_dimension;
  mps.impl->sv_threshold = sv_threshold;

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

std::complex<double> MatrixProductState::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

void MatrixProductState::print_mps() const {
  impl->print_mps();
}

DensityMatrix MatrixProductState::partial_trace(const std::vector<uint32_t>& qubits) const {
  return impl->partial_trace(qubits);
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
