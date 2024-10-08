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

class MatrixProductStateImpl {
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
      Eigen::Matrix2cd P;
      P << 1, 0, 0, 0;
      return P;
    }

    static Eigen::Matrix2cd one_projector() {
      Eigen::Matrix2cd P;
      P << 0, 0, 0, 1;
      return P;
    }



  public:
    MatrixProductStateImpl()=default;

    ~MatrixProductStateImpl()=default;

    MatrixProductStateImpl(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) 
    : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold) {
      std::random_device random_device;
      rng.seed(random_device());

      if (bond_dimension > 1u << num_qubits) {
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
    
    enum Pauli {
      I, X, Y, Z
    };

    struct PauliString {
      std::vector<Pauli> paulis;
      std::string to_string() const {
        std::string s;
        for (auto p : paulis) {
          if (p == Pauli::I) {
            s += "I";
          } else if (p == Pauli::X) {
            s += "X";
          } else if (p == Pauli::Y) {
            s += "Y";
          } else if (p == Pauli::Z) {
            s += "Z";
          }
        }

        return s;
      }
    };

    ITensor A(size_t i) const {
      // TODO: CHECK RIGHT NORMALIZATION
      auto Ai = tensors[i];

      Index right;
      Index left;
      if (i != 0) {
        left = findInds(Ai, fmt::format("a{}", i-1))[0];
      } else {
        left = Index(1, "a,Internal,Left");

        ITensor one(left);
        one.set(left=1, 1.0);
        Ai *= one;
      }

      if (i != num_qubits - 1) {
        right = findInds(Ai, fmt::format("a{}", i))[0];
      } else {
        right = Index(1, "a,Internal,Right");

        ITensor one(right);
        one.set(right=1, 1.0);
        Ai *= one*singular_values[i];
      }

      Index _left(dim(left), fmt::format("a{},Left,Internal", i));
      Index _right(dim(right), fmt::format("a{},Right,Internal", i));

      return Ai.swapInds(findInds(Ai, "Internal"), {_left, _right});
    }

    ITensor pauli_matrix(size_t i, Index i1, Index i2) const {
      Eigen::Matrix2cd c;
      if (i == 0) {
        c << 1.0, 0.0,
             0.0, 1.0;
      } else if (i == 1) {
        c << 0.0, 1.0,
             1.0, 0.0;
      } else if (i == 2) {
        c << 0.0, std::complex<double>(0.0, -1.0),
             std::complex<double>(0.0, 1.0), 0.0;
      } else if (i == 3) {
        c << 1.0, 0.0,
             0.0, -1.0;
      }

      return matrix_to_tensor(c, i1, i2);
    }

    std::pair<PauliString, double> sample_pauli() {
      std::vector<Pauli> p(num_qubits);
      double P = 1.0;

      Index i(1, "i");
      Index j(1, "j");

      ITensor L(i, j);
      L.set(i=1, j=1, 1.0);

      for (size_t k = 0; k < num_qubits-1; k++) {
        std::vector<double> probs(4);
        std::vector<ITensor> tensors(4);

        auto Ak = A(k);
        Index alpha_left = findInds(Ak, "Internal,Left")[0];
        L.replaceInds(inds(L), {alpha_left, prime(alpha_left)});

        Index s = external_indices[k];

        for (size_t p = 0; p < 4; p++) {
          auto sigma = pauli_matrix(p, s, prime(s));

          auto C = conj(Ak)*prime(Ak)*sigma;
          auto contraction = conj(C)*prime(C, 2, "Left");
          contraction = L * contraction * prime(L, 2);
          
          std::vector<size_t> inds;
          probs[p] = std::abs(eltC(contraction, inds))/2.0;
          tensors[p] = C;
        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        size_t a = dist(rng);

        p[k] = static_cast<Pauli>(a);
        P *= probs[a];
        L = tensors[a] / std::sqrt(2.0 * probs[a]);
      }

      return std::make_pair(PauliString{p}, P);
    }

    double stabilizer_renyi_entropy(size_t n, size_t num_samples = 100) {
      std::vector<double> samples(num_samples);
      double q = 0.0;
      if (n > 1) {
        for (size_t k = 0; k < num_samples; k++) {
          auto [pauli, p] = sample_pauli();
          samples[k] = std::pow(p, n - 1.0);
          q += std::pow(p, n - 1.0);
        }
      } else if (n == 1) {
        for (size_t k = 0; k < num_samples; k++) {
          auto [pauli, p] = sample_pauli();
          samples[k] = std::log(p);
          q += std::log(p);
        }
      } else {
        throw std::runtime_error("n = 0 stabilizer renyi entropy not implemented.");
      }

      q /= num_samples;

      return std::log(q)/(num_qubits*(1.0 - n)) - std::log(2);
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

      if (q1 == q2) {
        throw std::invalid_argument("Can only evolve gates on adjacent qubits (for now).");
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

    bool measure(uint32_t q, double r) {
      double prob_zero = measure_probability(q, 0);
      bool outcome = r < prob_zero;

      Eigen::Matrix2cd proj = outcome ? 
        MatrixProductStateImpl::one_projector()/std::sqrt(1.0 - prob_zero) :
        MatrixProductStateImpl::zero_projector()/std::sqrt(prob_zero);

      evolve(proj, q);

      Eigen::Matrix4cd id;
      id.setIdentity();

      // Propagate right
      for (uint32_t i = q; i < num_qubits - 1; i++) {
        if (dim(inds(singular_values[i])[0]) == 1) {
          break;
        }

        evolve(id, {i, i+1});
      }

      // Propagate left
      for (uint32_t i = q; i > 1; i--) {
        if (dim(inds(singular_values[i-1])[0]) == 1) {
          break;
        }

        evolve(id, {i-1, i});
      }

      return outcome;
    }
};

MatrixProductState::MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) : QuantumState(num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(num_qubits, bond_dimension, sv_threshold);
 }

MatrixProductState::~MatrixProductState()=default;

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

	uint32_t q = sorted_qubits.back();

	return impl->entropy(q);
}

double MatrixProductState::stabilizer_renyi_entropy(size_t n, size_t num_samples=100) const {
  return impl->stabilizer_renyi_entropy(n, num_samples);
}

void MatrixProductState::print_mps() const {
  impl->print_mps();
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
