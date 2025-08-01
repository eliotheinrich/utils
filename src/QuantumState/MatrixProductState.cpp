#include "QuantumStates.h"

#include <memory>
#include <sstream>
#include <random>

#include <fmt/ranges.h>
#include <itensor/all.h>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

#include <glaze/glaze.hpp>

using namespace itensor;

namespace glz::detail {
   template <>
   struct from<BEVE, ITensor> {
      template <auto Opts>
      static void op(ITensor& value, auto&&... args) {
        std::string str;
        read<BEVE>::op<Opts>(str, args...);
        std::istringstream stream(str);
        itensor::read(stream, value);
      }
   };

   template <>
   struct to<BEVE, ITensor> {
      template <auto Opts>
      static void op(const ITensor& value, auto&&... args) noexcept {
        std::stringstream data;
        itensor::write(data, value);
        write<BEVE>::op<Opts>(data.str(), args...);
      }
   };

   template <>
   struct from<BEVE, Index> {
      template <auto Opts>
      static void op(Index& value, auto&&... args) {
        std::string str;
        read<BEVE>::op<Opts>(str, args...);
        std::istringstream stream(str);
        itensor::read(stream, value);
      }
   };

   template <>
   struct to<BEVE, Index> {
      template <auto Opts>
      static void op(const Index& value, auto&&... args) noexcept {
        std::stringstream data;
        itensor::write(data, value);
        write<BEVE>::op<Opts>(data.str(), args...);
      }
   };
}

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

ITensor pauli_tensor(Pauli p, Index i1, Index i2) {
  if (p == Pauli::I) {
    return matrix_to_tensor(gates::I::value.asDiagonal(), {i1}, {i2});
  } else if (p == Pauli::X) {
    return matrix_to_tensor(gates::X::value, {i1}, {i2});
  } else if (p == Pauli::Y) {
    return matrix_to_tensor(gates::Y::value, {i1}, {i2});
  } else if (p == Pauli::Z) {
    return matrix_to_tensor(gates::Z::value.asDiagonal(), {i1}, {i2});
  }

  throw std::runtime_error("Invalid Pauli index.");
}

ITensor projection_tensor(bool z, Index i1, Index i2) {
  static const Eigen::Matrix2cd P0 = (Eigen::Matrix2cd() << 1.0, 0.0, 0.0, 0.0).finished();
  static const Eigen::Matrix2cd P1 = (Eigen::Matrix2cd() << 0.0, 0.0, 0.0, 1.0).finished();
  if (z) {
    return matrix_to_tensor(P1, {i1}, {i2});
  } else {
    return matrix_to_tensor(P0, {i1}, {i2});
  }
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

class MatrixProductStateImpl {
  friend class MatrixProductState;
  friend class MatrixProductMixedStateImpl;

  private:
  public:
		std::vector<ITensor> tensors;
    ITensor left_environment_tensor;
    ITensor right_environment_tensor;

    Index left_boundary_index;
    Index right_boundary_index;

		std::vector<Index> external_indices;
		std::vector<Index> internal_indices;

    uint32_t left_ortho_lim;
    uint32_t right_ortho_lim;

    std::vector<double> log;
    int debug_level;
    int orthogonality_level;

    uint32_t num_qubits;
    uint32_t bond_dimension;
    double sv_threshold;

    struct glaze {
      using T = MatrixProductStateImpl;
      static constexpr auto value = glz::object(
        &T::num_qubits,
        &T::bond_dimension,
        &T::sv_threshold,
        &T::tensors,
        &T::left_environment_tensor,
        &T::right_environment_tensor,
        &T::left_boundary_index,
        &T::right_boundary_index,
        &T::internal_indices,
        &T::external_indices,
        &T::left_ortho_lim,
        &T::right_ortho_lim,
        &T::log,
        &T::debug_level,
        &T::orthogonality_level
      );
    };

    MatrixProductStateImpl()=default;
    ~MatrixProductStateImpl()=default;

    MatrixProductStateImpl(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) 
      : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold), left_ortho_lim(0), right_ortho_lim(0),
        debug_level(0), orthogonality_level(1) {
        if (sv_threshold < 1e-15) {
          throw std::runtime_error("sv_threshold must be finite ( > 0) or else the MPS may be numerically unstable.");
        }

        if ((bond_dimension > 1u << num_qubits) && (num_qubits < 32)) {
          bond_dimension = 1u << num_qubits;
        }

        if (num_qubits < 1) {
          throw std::invalid_argument("Number of qubits must be > 1 for MPS simulator.");
        }

        log = {};

        left_boundary_index = Index(1, "Internal,LEdge");
        left_environment_tensor = delta(left_boundary_index, prime(left_boundary_index));
        right_boundary_index = Index(1, "Internal,REdge");
        right_environment_tensor = delta(right_boundary_index, prime(right_boundary_index));

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

        // Setting left boundary tensor
        tensor = ITensor(left_boundary_index, internal_idx(0), external_idx(0));
        tensor.set(1, 1, 1, 1.0);
        tensors.push_back(tensor);

        // Setting bulk tensors
        for (uint32_t q = 1; q < num_qubits - 1; q++) {
          tensor = ITensor(internal_idx(q - 1), internal_idx(q), external_idx(q));
          tensor.set(1, 1, 1, 1.0);
          tensors.push_back(tensor);
        }

        // Setting right boundary tensor
        tensor = ITensor(internal_idx(num_qubits - 2), right_boundary_index, external_idx(num_qubits - 1));
        tensor.set(1, 1, 1, 1.0);
        tensors.push_back(tensor);
      }

    MatrixProductStateImpl(const MatrixProductStateImpl& other) : MatrixProductStateImpl(other.num_qubits, other.bond_dimension, other.sv_threshold) {
      tensors = other.tensors;

      left_environment_tensor = other.left_environment_tensor;
      right_environment_tensor = other.right_environment_tensor;

      left_boundary_index = other.left_boundary_index;
      right_boundary_index = other.right_boundary_index;
      internal_indices = other.internal_indices;
      external_indices = other.external_indices;

      left_ortho_lim = other.left_ortho_lim;
      right_ortho_lim = other.right_ortho_lim;

      log = other.log;
      debug_level = other.debug_level;
      orthogonality_level = other.orthogonality_level;
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
          u_inds.push_back(vidal_mps.internal_idx(i - 2));
        }

        if (i != mps.length() - 1) {
          v_inds.push_back(linkIndex(mps, i + 1));
        }


        auto M = V*mps(i + 1);

        std::tie(U, S, V) = svd(M, u_inds, v_inds, {
          "SVDMethod=","gesvd",
          "Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
          "LeftTags=",fmt::format("tmp,Internal,n={},",i-1),
          "RightTags=",fmt::format("Internal,n={}",i-1)
        });

        auto ext = siteIndex(mps, i);
        U.replaceTags(ext.tags(), vidal_mps.external_idx(i - 1).tags());
        if (i != 1) {
          U.replaceTags(linkIndex(mps, i-1).tags(), tags(vidal_mps.internal_idx(i-1)));
        }

        if (i != num_qubits-1) {
          U.replaceTags(linkIndex(mps, i).tags(), tags(vidal_mps.internal_idx(i)));
        }

        // TODO check
        vidal_mps.tensors[i-1] = U * S;

        vidal_mps.external_indices[i-1] = findInds(U, "External")[0];
        vidal_mps.internal_indices[i-1] = findInds(V, "Internal")[0];
      }

      V.replaceTags(siteIndex(mps, num_qubits).tags(), vidal_mps.external_idx(num_qubits - 1).tags());
      vidal_mps.external_indices[num_qubits - 1] = findInds(V, "External")[0];
      vidal_mps.tensors[num_qubits - 1] = V;


      // Add boundary indices
      ITensor one;

      one = ITensor(vidal_mps.left_boundary_index);
      one.set(1, 1.0);
      vidal_mps.tensors[0] *= one;

      one = ITensor(vidal_mps.right_boundary_index);
      one.set(1, 1.0);
      vidal_mps.tensors[num_qubits - 1] *= one;

      vidal_mps.set_left_ortho_lim(0);
      vidal_mps.set_right_ortho_lim(num_qubits - 1);

      return vidal_mps;
    }

    void set_debug_level(int i) {
      if (i > 2) {
        throw std::runtime_error("Debug level must be one of [0, 1, 2].");
      }
      debug_level = i;
    }

    void set_orthogonality_level(int i) {
      if (i > 1) {
        throw std::runtime_error("Orthogonality level must be one of [0, 1].");
      }
      orthogonality_level = i;
    }

    std::pair<uint32_t, uint32_t> get_boundaries(const Qubits& qubits) {
      if (qubits.size() == num_qubits) {
        throw std::runtime_error("Cannot trace over every qubit of MPS.");
      }

      Qubits sorted(qubits.begin(), qubits.end());
      std::sort(sorted.begin(), sorted.end());

      uint32_t q1 = 0;
      while (q1 == sorted[q1] && q1 < num_qubits) {
        q1++;
      }

      uint32_t q2 = num_qubits - 1;
      size_t k = sorted.size() - 1;
      while (q2 == sorted[k] && q2 > 0) {
        k--;
        q2--;
      }
      q2++;

      size_t left_width = q1;
      size_t right_width = num_qubits - q2;

      if (right_width + left_width != qubits.size()) {
        throw std::runtime_error(fmt::format("Provided qubits {} are not exclusively at the boundary. Cannot perform MatrixProductState.partial_trace.", qubits));
      }

      return {q1, q2};
    }

    MatrixProductStateImpl partial_trace(const Qubits& qubits) {
      if (qubits.size() == 0) {
        return MatrixProductStateImpl(*this);
      }

      auto [q1, q2] = get_boundaries(qubits);
      orthogonalize(q1, q2);

      size_t remaining_qubits = num_qubits - qubits.size();

      std::vector<ITensor> tensors_;

      std::vector<Index> internal_indices_;
      std::vector<Index> external_indices_;

      ITensor left_environment_tensor_ = build_left_environment_tensor(q1);
      ITensor right_environment_tensor_ = build_right_environment_tensor(q2);

      for (size_t q = q1; q < q2; q++) {
        tensors_.push_back(tensors[q]);
      }

      uint32_t lqb = remaining_qubits - 1;
      
      // Fix left environment index/tensor
      Index left_ = commonIndex(left_environment_tensor_, tensors_[0]);
      Index left_boundary_index_ = Index(dim(left_), "Internal,LEdge");
      try {
        left_environment_tensor_.replaceInds({left_, prime(left_)}, {left_boundary_index_, prime(left_boundary_index_)});
      } catch (const ITError& error) {
        left_environment_tensor_ = toDense(left_environment_tensor_);
        left_environment_tensor_.replaceInds({left_, prime(left_)}, {left_boundary_index_, prime(left_boundary_index_)});
      }
      tensors_[0].replaceInds({left_}, {left_boundary_index_});
      
      // Fix right environment index/tensor
      Index right_ = commonIndex(right_environment_tensor_, tensors_[lqb]);
      Index right_boundary_index_ = Index(dim(right_), "Internal,REdge");
      try {
        right_environment_tensor_.replaceInds({right_, prime(right_)}, {right_boundary_index_, prime(right_boundary_index_)}); 
      } catch (const ITError& error) {
        right_environment_tensor_ = toDense(right_environment_tensor_);
        right_environment_tensor_.replaceInds({right_, prime(right_)}, {right_boundary_index_, prime(right_boundary_index_)}); 
      }
      tensors_[lqb].replaceInds({right_}, {right_boundary_index_});

      // Correct internal indices
      for (size_t j = 0; j < remaining_qubits - 1; j++) {
        size_t i = j + q1;
        auto internal = findIndex(tensors_[j], fmt::format("Internal,n={}",i));
        auto internal_ = Index(dim(internal), fmt::format("Internal,n={}",j));
        tensors_[j].replaceInds({internal}, {internal_});
        tensors_[j+1].replaceInds({internal}, {internal_});
        internal_indices_.push_back(internal_);
      }

      // Correct external indices
      for (size_t q = 0; q < remaining_qubits; q++) {
        Index external = findIndex(tensors_[q], "External");
        Index external_(dim(external), fmt::format("External,i={}", q));
        tensors_[q].replaceInds({external}, {external_});
        external_indices_.push_back(external_);
      }

      MatrixProductStateImpl mps(remaining_qubits, bond_dimension, sv_threshold);
      
      mps.tensors = tensors_;

      mps.left_environment_tensor = left_environment_tensor_;
      mps.right_environment_tensor = right_environment_tensor_;

      mps.left_boundary_index = left_boundary_index_;
      mps.right_boundary_index = right_boundary_index_;

      mps.external_indices = external_indices_;
      mps.internal_indices = internal_indices_;

      mps.left_ortho_lim = 0;
      mps.right_ortho_lim = remaining_qubits - 1;
      mps.debug_level = debug_level;
      mps.orthogonality_level = orthogonality_level; 

      return mps;
    }

    Index internal_idx(size_t i) const {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot retrieve internal index for i = {}.", i));
      }

      return internal_indices[i];

    }

    Index external_idx(size_t i) const {
      if (i >= num_qubits) {
        throw std::runtime_error(fmt::format("Cannot retrieve external index for i = {}.", i));
      }

      return external_indices[i];
    }

    std::string to_string() {
      if (is_pure_state()) {
        Statevector psi(coefficients_pure());
        return psi.to_string();
      } else {
        DensityMatrix rho(coefficients_mixed());
        return rho.to_string();
      }
    }

    void print_mps(bool print_data=false) const {
      auto print_tensor = [&](const ITensor& tensor) {
        if (print_data) {
          PrintData(tensor);
        } else {
          print(tensor);
        }
      };

      print_tensor(left_environment_tensor);
      print_tensor(tensors[0]);

      for (size_t q = 0; q < num_qubits - 1; q++) {
        print_tensor(tensors[q+1]);
      }

      print_tensor(right_environment_tensor);
    }

    void set_left_ortho_lim(uint32_t q) {
      left_ortho_lim = std::min(left_ortho_lim, q);
    }

    void set_right_ortho_lim(uint32_t q) {
      if (q == num_qubits) {
        throw std::runtime_error("Why has this happened\n");
      }
      right_ortho_lim = std::max(right_ortho_lim, q);
    }

    void left_orthogonalize(uint32_t q) {
      if (q == 0 || orthogonality_level == 0) {
        return;
      }

      while (left_ortho_lim < q) {
        svd_bond(left_ortho_lim++, nullptr, false, InternalDir::Right);
        if (left_ortho_lim > right_ortho_lim) {
          right_ortho_lim++;
        }
      }
    }

    void right_orthogonalize(uint32_t q) {
      if (q == num_qubits - 1 || orthogonality_level == 0) {
        return;
      }

      while (right_ortho_lim > q) {
        svd_bond(--right_ortho_lim, nullptr, false, InternalDir::Left);
        if (right_ortho_lim < left_ortho_lim) {
          left_ortho_lim--;
        }
      }
    }

    void orthogonalize(size_t q) {
      if (q > num_qubits - 1 || q < 0) {
        throw std::runtime_error(fmt::format("Cannot move orthogonality center of state with {} qubits to site {}\n", num_qubits, q));
      }

      left_orthogonalize(q);
      right_orthogonalize(q);
    }

    void orthogonalize(size_t q1, size_t q2) {
      left_orthogonalize(q1);
      right_orthogonalize(q2 - 1);
    }

    bool is_orthogonal() const {
      if (orthogonality_level == 0) {
        return true;
      } else {
        return left_ortho_lim >= right_ortho_lim;
      }
    }

    std::vector<double> singular_values_to_vector(size_t i) {
      if (i >= internal_indices.size()) {
        throw std::runtime_error(fmt::format("Cannot retrieve singular values in index {} for MPS with {} qubits.", i, num_qubits));
      }

      orthogonalize(i);
      ITensor singular_values = svd_bond(i);
      uint32_t N = dim(inds(singular_values)[0]);

      std::vector<double> sv(bond_dimension, 0.0);
      for (uint32_t j = 0; j < N; j++) {
        std::vector<uint32_t> assignments{j + 1, j + 1};
        sv[j] = elt(singular_values, assignments);
      }

      return sv;
    }

    std::pair<std::vector<size_t>, std::vector<std::complex<double>>> get_tensor(uint32_t q) {
      orthogonalize(q);

      ITensor T = tensors[q];

      Index ext = external_idx(q);

      Index left_idx;
      if (q == 0) {
        left_idx = left_boundary_index;
      } else {
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

      std::vector<size_t> shape = {d1, d2, 2};
      std::vector<std::complex<double>> values(d1*d2*2);

      size_t k = 0;
      for (uint32_t a1 = 0; a1 < d1; a1++) {
        for (uint32_t a2 = 0; a2 < d2; a2++) {
          for (uint32_t i = 0; i < 2; i++) {
            values[k++] = eltC(T, left_idx=a1+1, right_idx=a2+1, ext=i+1);
          }
        }
      }

      return std::make_pair(shape, values);
    }

    double entanglement(uint32_t q, uint32_t index) {
      if (q < 0 || q > num_qubits) {
        throw std::invalid_argument("Invalid qubit passed to MatrixProductState.entanglement; must have 0 <= q <= num_qubits.");
      }

      if (q == 0 || q == num_qubits) {
        return 0.0;
      }

      orthogonalize(q);
      ITensor sv = svd_bond(q - 1, nullptr, false, InternalDir::Right);
      assert_state_valid(fmt::format("State invalid after applying entanglement({})\n", q));

      int d = dim(inds(sv)[0]);

      std::vector<double> sv_squared(d);
      for (int i = 0; i < d; i++) {
        sv_squared[i] = std::pow(elt(sv, i+1, i+1), 2);
      }

      return renyi_entropy(index, sv_squared);
    }

    ITensor left_boundary_tensor(size_t i) const {
      if (i == 0) {
        return left_environment_tensor;
      }

      Index idx = internal_idx(i - 1);
      return delta(idx, prime(idx));
    }

    ITensor build_left_environment_tensor(size_t i, const std::vector<ITensor>& external_tensors) const {
      ITensor L = left_environment_tensor;
      extend_left_environment_tensor(L, 0, i, external_tensors);
      return L;
    }

    ITensor build_left_environment_tensor(size_t i) const {
      std::vector<ITensor> external_tensors = get_deltas_between(0, i);
      return build_left_environment_tensor(i, external_tensors);
    }

    void extend_left_environment_tensor(ITensor& L, uint32_t i1, uint32_t i2, const std::vector<ITensor>& external_tensors) const {
      if (i1 != i2) {
        L = partial_contraction(i1, i2, &external_tensors, &L, nullptr);
      }
    }

    void extend_left_environment_tensor(ITensor& L, uint32_t i1, uint32_t i2) const {
      std::vector<ITensor> external_tensors = get_deltas_between(i1, i2);
      extend_left_environment_tensor(L, i1, i2, external_tensors);
    }

    ITensor right_boundary_tensor(size_t i) const {
      if (i == num_qubits) {
        return right_environment_tensor;
      }
      Index idx = internal_idx(i - 1);
      return delta(idx, prime(idx));
    }

    ITensor build_right_environment_tensor(size_t i, const std::vector<ITensor>& external_tensors) const {
      ITensor R = right_environment_tensor;
      extend_right_environment_tensor(R, num_qubits, i, external_tensors);
      return R;
    }

    ITensor build_right_environment_tensor(size_t i) const {
      std::vector<ITensor> external_tensors = get_deltas_between(i, num_qubits);
      return build_right_environment_tensor(i, external_tensors);
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t i1, uint32_t i2, const std::vector<ITensor>& external_tensors) const {
      if (i1 != i2) {
        R = partial_contraction(i2, i1, &external_tensors, nullptr, &R);
      }
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t i1, uint32_t i2) const {
      std::vector<ITensor> external_tensors = get_deltas_between(i2, i1);
      extend_right_environment_tensor(R, i1, i2, external_tensors);
    }

    ITensor partial_contraction(
        size_t i1, size_t i2, 
        const std::vector<ITensor>* external_tensors, const ITensor* L, const ITensor* R
      ) const {

      // If a right environment is provided but not a left, contract from right-to-left instead of left-to-right
      bool left = !(R && !L);

      ITensor contraction;

      size_t k = 0;

      auto advance_contraction = [&](ITensor& C, size_t q) {
        ITensor tensor = tensors[q];

        if (!C) {
          C = tensor;
        } else {
          C *= tensor;
        }

        if (external_tensors != nullptr && k < external_tensors->size()) {
          size_t p = left ? k : (external_tensors->size() - 1 - k);
          k++;
          C *= (*external_tensors)[p];
        }
        C *= conj(prime(tensor));
      };

      const ITensor* T1 = left ? L : R;
      const ITensor* T2 = left ? R : L;

      if (T1 != nullptr) {
        contraction = *T1;
      }

      size_t width = i2 - i1;

      for (size_t j = 0; j < width; j++) {
        size_t n = left ? (i1 + j) : (i2 - j - 1);
        advance_contraction(contraction, n);
      }

      if (T2 != nullptr) {
        contraction *= (*T2);
      }

      return contraction;
    }

    std::complex<double> partial_expectation(const Eigen::MatrixXcd& m, uint32_t i1, uint32_t i2, const ITensor& L, const ITensor& R) const {
      std::vector<Index> idxs;
      std::vector<Index> idxs_;

      for (size_t q = i1; q < i2; q++) {
        Index idx = external_idx(q);
        idxs.push_back(idx);
        idxs_.push_back(prime(idx));
      }

      std::vector<ITensor> mtensor = {matrix_to_tensor(m, idxs_, idxs)};
      ITensor contraction = partial_contraction(i1, i2, &mtensor, &L, &R);
      return tensor_to_scalar(contraction);
    }

    std::complex<double> partial_expectation(const std::vector<ITensor>& operators, uint32_t i1, uint32_t i2, const ITensor& L, const ITensor& R) const {
      assert_state_valid(fmt::format("State invalid while trying to call partial_expectation\n"));
      ITensor contraction = partial_contraction(i1, i2, &operators, &L, &R);
      return tensor_to_scalar(contraction);
    }

    std::pair<ITensor, ITensor> get_boundary_tensors(size_t i1, size_t i2) {
      if (orthogonality_level == 0) {
        return {build_left_environment_tensor(i1), build_right_environment_tensor(i2)};
      } else {
        orthogonalize(i1, i2);
        return {left_boundary_tensor(i1), right_boundary_tensor(i2)};
      }
    }

    std::complex<double> expectation(const PauliString& p) {
      if (p.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Provided PauliString has {} qubits but MatrixProductState has {} qubits.", p.num_qubits, num_qubits));
      }

      auto qubit_range = p.support_range();

      // Pauli is proportional to I; return sign.
      if (qubit_range == std::nullopt) {
        return p.sign();
      }

      auto [q1, q2] = qubit_range.value();

      auto [L, R] = get_boundary_tensors(q1, q2);
      orthogonalize(q1);

      std::vector<ITensor> paulis;
      for (size_t q = q1; q < q2; q++) {
        Index idx = external_idx(q);
        paulis.push_back(pauli_tensor(p.to_pauli(q), prime(idx), idx));
      }

      return p.sign() * partial_expectation(paulis, q1, q2, L, R);
    }

    double expectation(const BitString& bits, std::optional<QubitSupport> support) {
      QubitSupport _support;
      if (support) {
        _support = support.value();
      } else {
        _support = std::make_pair(0, num_qubits);
      }

      QubitInterval interval = to_interval(_support);

      uint32_t q1, q2;
      if (interval) {
        std::tie(q1, q2) = interval.value();
      } else {
        return 1.0;
      }

      auto [L, R] = get_boundary_tensors(q1, q2);

      std::vector<ITensor> operators;

      Qubits qubits = to_qubits(_support);
      std::set<uint32_t> qubits_set(qubits.begin(), qubits.end());
      for (size_t q = q1; q < q2; q++) {
        Index idx = external_idx(q);
        ITensor op;
        if (qubits_set.contains(q)) {
          op = projection_tensor(bits.get(q), prime(idx), idx);
        } else {
          op = matrix_to_tensor(Eigen::Matrix2cd::Identity(), {prime(idx)}, {idx});
        }

        operators.push_back(op);
      }

      return partial_expectation(operators, q1, q2, L, R).real();
    }

    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) {
      if (qubits.size() == 0) {
        throw std::runtime_error("Cannot compute expectation over 0 qubits.");
      }

      size_t r = m.rows();
      size_t c = m.cols();
      size_t n = qubits.size();
      if (r != c || (1u << n != r)) {
        throw std::runtime_error(fmt::format("Passed observable has dimension {}x{}, provided {} sites.", r, c, n));
      }

      if (!support_contiguous(qubits)) {
        throw std::runtime_error(fmt::format("Provided sites {} are not contiguous.", qubits));
      }

      auto [q1, q2] = to_interval(qubits).value();
      
      auto [L, R] = get_boundary_tensors(q1, q2);

      return partial_expectation(m, q1, q2, L, R);
    }

    std::vector<BitAmplitudes> sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) {
      int level = orthogonality_level;
      orthogonalize(0);
      set_orthogonality_level(0);

      std::vector<BitAmplitudes> samples;
      
      for (size_t i = 0; i < num_samples; i++) {
        double p = 1.0;
        BitString bits(num_qubits);
        ITensor L = left_environment_tensor;

        for (size_t q = 0; q < num_qubits; q++) {
          Index ext = external_idx(q);
          std::vector<ITensor> M = {projection_tensor(0, prime(ext), ext)};


          ITensor R = right_boundary_tensor(q+1);
          ITensor L0 = partial_contraction(q, q+1, &M, &L, nullptr);

          double p0 = std::abs(tensor_to_scalar(L0 * R))/p;
          
          bool v = (randf() >= p0);
          bits.set(q, v);
          if (v) {
            M = {projection_tensor(1, prime(ext), ext)};
            ITensor L1 = partial_contraction(q, q+1, &M, &L, nullptr);
            L = L1;
            p *= 1.0 - p0;
          } else {
            L = L0;
            p *= p0;
          }
        }
        
        std::vector<double> amplitudes{p};
        for (const auto& support : supports) {
          amplitudes.push_back(expectation(bits, support));
        }

        samples.push_back({bits, amplitudes});
      }

      set_orthogonality_level(level);
      return samples;
    }

    std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& supports, size_t num_samples) {
      int level = orthogonality_level;
      orthogonalize(0);
      set_orthogonality_level(0);

      std::vector<PauliAmplitudes> samples(num_samples);
      std::minstd_rand rng(randi());

      for (size_t k = 0; k < num_samples; k++) {
        std::vector<Pauli> p(num_qubits);
        double P = 1.0;

        ITensor L = left_environment_tensor;

        for (size_t q = 0; q < num_qubits; q++) {
          std::vector<double> probs(4);
          std::vector<ITensor> pauli_tensors(4);

          Index s = external_indices[q];

          for (size_t p = 0; p < 4; p++) {
            std::vector<ITensor> sigma = {pauli_tensor(static_cast<Pauli>(p), prime(s), s)};

            auto C = partial_contraction(q, q+1, &sigma, &L, nullptr);

            auto contraction = conj(C) * C / 2.0;

            std::vector<size_t> inds;
            double prob = std::abs(eltC(contraction, inds));
            probs[p] = prob;
            pauli_tensors[p] = C / std::sqrt(2.0 * prob);
          }

          std::discrete_distribution<> dist(probs.begin(), probs.end());
          size_t a = dist(rng);

          p[q] = static_cast<Pauli>(a);
          P *= probs[a];
          L = pauli_tensors[a];
        }

        PauliString pauli(p);

        double t = std::sqrt(P*std::pow(2.0, num_qubits));
        std::vector<double> amplitudes{t};
        // TODO make this more efficient with partial contractions
        for (const auto& support : supports) {
          PauliString ps = pauli.substring(support, false);
          amplitudes.push_back(std::abs(expectation(ps)));
        }

        samples[k] = {pauli, amplitudes};
      }

      set_orthogonality_level(level);
      return samples;
    }

    std::vector<double> process_bipartite_pauli_samples(const std::vector<PauliAmplitudes>& pauli_samples) {
      int level = orthogonality_level; 
      size_t N = num_qubits/2 - 1;
      size_t num_samples = pauli_samples.size();
      std::vector<std::vector<double>> samplesA(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesB(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesAB(N, std::vector<double>(num_samples));

      for (size_t j = 0; j < num_samples; j++) {
        auto const [P, t] = pauli_samples[j];

        set_orthogonality_level(level);
        orthogonalize(0);
        set_orthogonality_level(0);

        std::vector<double> tA(N);
        ITensor L = left_environment_tensor;
        for (size_t q = 0; q < N; q++) {
          Index idx = external_idx(q);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q), prime(idx), idx)};
          L = partial_contraction(q, q+1, &p, &L, nullptr);

          ITensor contraction = L * right_boundary_tensor(q+1);
          samplesA[q][j] = std::abs(tensor_to_scalar(contraction));
        }

        std::vector<double> tB(N);
        std::vector<ITensor> paulis;
        for (size_t q = num_qubits/2; q < num_qubits; q++) {
          Index idx = external_idx(q);
          paulis.push_back(pauli_tensor(P.to_pauli(q), prime(idx), idx));
        }

        set_orthogonality_level(level);
        orthogonalize(num_qubits - 1);
        set_orthogonality_level(0);

        size_t i = num_qubits/2;
        ITensor R = partial_contraction(i, num_qubits, &paulis, nullptr, &right_environment_tensor);
        for (size_t n = 0; n < N; n++) {
          uint32_t q = num_qubits/2 - n;
          Index idx = external_idx(q - 1);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q-1), prime(idx), idx)};
          R = partial_contraction(q-1, q, &p, nullptr, &R);

          ITensor contraction = left_boundary_tensor(q-1) * R;
          samplesB[N - 1 - n][j] = std::abs(tensor_to_scalar(contraction));
        }

        set_orthogonality_level(level);

        for (size_t n = 0; n < N; n++) {
          samplesAB[n][j] = t[0];
        }
      }

      std::vector<double> magic(N);
      for (size_t n = 0; n < N; n++) {
        magic[n] = MatrixProductState::calculate_magic_mutual_information_from_samples2(samplesAB[n], samplesA[n], samplesB[n]);
      }

      return magic;
    }

    std::vector<std::vector<double>> process_bipartite_bit_samples(const std::vector<size_t>& renyi_indices, const std::vector<BitAmplitudes>& bit_samples) {
      int level = orthogonality_level;
      size_t N = num_qubits/2 - 1;
      size_t num_samples = bit_samples.size();
      std::vector<std::vector<double>> samplesA(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesB(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesAB(N, std::vector<double>(num_samples));

      for (size_t j = 0; j < num_samples; j++) {
        auto const [b, t] = bit_samples[j];

        set_orthogonality_level(level);
        orthogonalize(0);
        set_orthogonality_level(0);

        std::vector<double> tA(N);
        ITensor L = left_environment_tensor;
        for (size_t q = 0; q < N; q++) {
          Index idx = external_idx(q);
          std::vector<ITensor> p = {projection_tensor(b.get(q), prime(idx), idx)};
          L = partial_contraction(q, q+1, &p, &L, nullptr);

          ITensor contraction = L * right_boundary_tensor(q+1);
          samplesA[q][j] = std::abs(tensor_to_scalar(contraction));
        }

        std::vector<double> tB(N);
        std::vector<ITensor> projectors;
        for (size_t q = num_qubits/2; q < num_qubits; q++) {
          Index idx = external_idx(q);
          projectors.push_back(projection_tensor(b.get(q), prime(idx), idx));
        }

        set_orthogonality_level(level);
        orthogonalize(num_qubits - 1);
        set_orthogonality_level(0);

        size_t i = num_qubits/2;
        ITensor R = partial_contraction(i, num_qubits, &projectors, nullptr, &right_environment_tensor);
        for (size_t n = 0; n < N; n++) {
          uint32_t q = num_qubits/2 - n;
          Index idx = external_idx(q-1);
          std::vector<ITensor> p = {projection_tensor(b.get(q-1), prime(idx), idx)};
          R = partial_contraction(q-1, q, &p, nullptr, &R);

          ITensor contraction = left_boundary_tensor(q-1) * R;
          samplesB[N - 1 - n][j] = std::abs(tensor_to_scalar(contraction));
        }

        set_orthogonality_level(level);

        for (size_t n = 0; n < N; n++) {
          samplesAB[n][j] = t[0];
        }
      }

      std::vector<std::vector<double>> data(renyi_indices.size(), std::vector<double>(N));
      for (size_t i = 0; i < renyi_indices.size(); i++) {
        for (size_t n = 0; n < N; n++) {
          data[i][n] = estimate_mutual_renyi_entropy(renyi_indices[i], samplesAB[n], samplesA[n], samplesB[n], 2);
        }
      }

      return data;
    }

    bool is_pure_state() const {
      return dim(left_boundary_index) == 1 && dim(right_boundary_index) == 1;
    }

    Eigen::MatrixXcd coefficients_mixed() {
      if (num_qubits > 15) {
        throw std::runtime_error("Cannot generate coefficients for n > 15 qubits.");
      }

      auto contraction = partial_contraction(0, num_qubits, nullptr, &left_environment_tensor, &right_environment_tensor);

      size_t s = 1u << num_qubits; 
      Eigen::MatrixXcd data = Eigen::MatrixXcd::Zero(s, s);

      for (size_t z1 = 0; z1 < s; z1++) {
        for (size_t z2 = 0; z2 < s; z2++) {
          std::vector<IndexVal> assignments(2*num_qubits);
          for (size_t j = 0; j < num_qubits; j++) {
            const Index& idx = external_idx(j);
            assignments[2*j + 1] = (idx = ((z1 >> j) & 1u) + 1);
            assignments[2*j] = (prime(idx) = ((z2 >> j) & 1u) + 1);
          }

          data(z1, z2) = eltC(contraction, assignments);
        }
      }

      return data;
    }

    Eigen::VectorXcd coefficients_pure() {
      if (num_qubits > 31) {
        throw std::runtime_error("Cannot generate coefficients for n > 31 qubits.");
      }

      if (!is_pure_state()) {
        throw std::runtime_error("Cannot calculate coefficients for mixed MatrixProductState.");
      }
      
      orthogonalize(0);

      ITensor C = tensors[0];

      for (uint32_t q = 0; q < num_qubits - 1; q++) {
        C *= tensors[q+1];
      }

      C *= delta(left_boundary_index, right_boundary_index);

      std::vector<uint32_t> indices(1u << num_qubits);
      std::iota(indices.begin(), indices.end(), 0);

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

    void swap(uint32_t q1, uint32_t q2) {
      evolve(gates::SWAP::value, {q1, q2});
    }

    enum InternalDir {
      Left, Right
    };

    std::string to_string(InternalDir dir) const {
      if (dir == InternalDir::Left) { return "Left"; } else { return "Right"; }
    }

    ITensor svd_bond(uint32_t q, ITensor* T=nullptr, bool truncate=true, InternalDir dir=InternalDir::Left) {
      size_t q1 = q;
      size_t q2 = q + 1;

      ITensor theta = tensors[q1] * tensors[q2];
      if (T) {
        theta = noPrime(theta * (*T));
      }

      std::vector<Index> u_inds{external_idx(q1)};
      std::vector<Index> v_inds{external_idx(q2)};

      Index left;
      if (q1 != 0) {
        left = internal_idx(q1 - 1);
      } else {
        left = left_boundary_index;
      }
      u_inds.push_back(left);

      Index right;
      if (q2 != num_qubits - 1) {
        right = internal_idx(q2);
      } else {
        right = right_boundary_index;
      }
      v_inds.push_back(right);

      double threshold = truncate ? sv_threshold : 1e-15;

      std::string left_tags = fmt::format("tmp,Internal,n={}", q1);
      std::string right_tags = fmt::format("Internal,n={}", q1);
      if (dir == InternalDir::Right) {
        std::swap(left_tags, right_tags);
      }

      // TODO can this be removed?
      theta = apply(theta, [](Cplx c) { 
        double mask = std::abs(c) >= 1e-14;
        return Cplx(c.real() * mask, c.imag() * mask);
      });

      auto [U, S, V] = svd(theta, u_inds, v_inds, {
        "SVDMethod=","gesvd",
        "Cutoff=",threshold,"MaxDim=",bond_dimension,
        "LeftTags=",left_tags,
        "RightTags=",right_tags
      });

      double truncerr = sqr(norm(U*S*V - theta)/norm(theta));
      log.push_back(truncerr);

      // Renormalize singular values
      size_t N = dim(inds(S)[0]);
      double d = 0.0;
      for (uint32_t p = 1; p <= N; p++) {
        std::vector<uint32_t> assignment = {p, p};
        double c = elt(S, assignment);
        d += c*c;
      }
      S /= std::sqrt(d);

      // SHOULD BE IDENTITY
      //PrintData(U * conj(prime(U, commonIndex(U, S))));
      //PrintData(V * conj(prime(V, commonIndex(V, S))));

      if (dir == InternalDir::Left) {
        U *= S;
      } else {
        V *= S;
      }

      internal_indices[q1] = commonIndex(U, V);
      tensors[q1] = U;
      tensors[q2] = V;

      return S;
    }

    void evolve(const Eigen::Matrix2cd& gate, uint32_t q) {
      auto i = external_indices[q];
      auto ip = prime(i);
      ITensor tensor = matrix_to_tensor(gate, {ip}, {i});
      tensors[q] = noPrime(tensors[q]*tensor);

      std::stringstream stream;
      stream << "Error after applying gate \n" << gate << fmt::format("\n to qubit {}.", q);
      assert_state_valid(stream.str());
    }

    void evolve(const Eigen::Matrix4cd& gate, uint32_t q1_, uint32_t q2_) {
      uint32_t q1 = std::min(q1_, q2_);
      uint32_t q2 = std::max(q1_, q2_);

      orthogonalize(q1);

      if (q2 - q1 > 1) {
        for (size_t q = q1; q < q2 - 1; q++) {
          swap(q, q+1);
        }

        Qubits qbits{q2 - 1, q2};
        if (q1_ > q2_) {
          Eigen::Matrix4cd SWAP = gates::SWAP::value;
          evolve(SWAP * gate * SWAP, qbits);
        } else {
          evolve(gate, qbits);
        }

        // Backward swaps
        for (size_t q = q2 - 1; q > q1; q--) {
          swap(q-1, q);
        }

        return;
      }

      auto i1 = external_idx(q1_);
      auto i2 = external_idx(q2_);
      ITensor gate_tensor = matrix_to_tensor(gate, 
        {prime(i1), prime(i2)}, 
        {i1,        i2}
      );

      svd_bond(q1, &gate_tensor);
      set_left_ortho_lim(q1);
      set_right_ortho_lim(q1+1);

      std::stringstream stream;
      stream << "Error after applying gate \n" << gate << fmt::format("\n to qubits ({}, {}).", q1_, q2_);
      assert_state_valid(stream.str());
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

      uint32_t qmin = std::ranges::min(qubits);
      uint32_t qmax = qmin + qubits.size();
      orthogonalize(qmin, qmax);

      std::vector<uint32_t> inds(num_qubits);
      std::iota(inds.begin(), inds.end(), 0);
      std::vector<std::pair<uint32_t, uint32_t>> swaps;
      for (size_t i = 0; i < qubits.size(); i++) {
        uint32_t q = qubits[i];
        auto it = std::find(inds.begin(), inds.end(), q);
        uint32_t current_pos;
        if (it != inds.end()) {
          current_pos = std::distance(inds.begin(), it);
        } else {
          throw std::runtime_error("Error finding qubit.");
        }

        for (uint32_t qi = current_pos; qi >= qmin + i + 1; qi--) {
          std::swap(inds[qi], inds[qi-1]);
          swaps.push_back({qi, qi-1});
        }
      }

      for (size_t j = 0; j < swaps.size(); j++) {
        auto [q1, q2] = swaps[j];
        swap(q1, q2);
      }

      // Now, gate is applied to [qmin, ..., qmax]
      std::vector<Index> indices;
      std::vector<Index> indices_p;
      for (uint32_t q = qmin; q < qmax; q++) {
        indices.push_back(external_idx(q));
        indices_p.push_back(prime(external_idx(q)));
      }

      ITensor theta = tensors[qmin];
      for (size_t q = qmin+1; q < qmax; q++) {
        theta *= tensors[q];
      }

      theta = noPrime(theta * matrix_to_tensor(gate, indices_p, indices));
      reset_from_tensor(theta, std::make_pair(qmin, qmax));

      // Reverse swaps
      for (size_t j = swaps.size(); j > 0; j--) {
        auto [q1, q2] = swaps[j - 1];
        swap(q1, q2);
      }
    }

    void reset_from_tensor(const ITensor& tensor, QubitInterval support) {
      if (!support) {
        return;
      }

      auto [q1, q2] = support.value();
      ITensor c = tensor;

      Index left = left_boundary_index;
      if (q1 != 0) {
        left = internal_idx(q1 - 1);
      }
      
      Index right = right_boundary_index;
      if (q2 != num_qubits) {
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
          "LeftTags=",fmt::format("tmp,Internal,n={}", i),
          "RightTags=",fmt::format("Internal,n={}", i)
        });

        double truncerr = sqr(norm(U*S*V - c)/norm(c));
        log.push_back(truncerr);

        // Renormalize singular values
        size_t N = dim(inds(S)[0]);
        double d = 0.0;
        for (uint32_t p = 1; p <= N; p++) {
          std::vector<uint32_t> assignment = {p, p};
          double c = elt(S, assignment);
          d += c*c;
        }
        S /= std::sqrt(d);

        c = V;
        tensors[i] = U * S;

        left = commonIndex(V, S);
        internal_indices[i] = commonIndex(S, V);
      }

      tensors[q2 - 1] = c;

      set_left_ortho_lim(q1);
      set_right_ortho_lim(q2-1);

      assert_state_valid(fmt::format("Failed after reset_from_tensor"));
    }

    double trace() const {
      std::vector<ITensor> deltas = get_deltas_between(0, num_qubits);
      ITensor contraction = partial_contraction(0, num_qubits, &deltas, &left_environment_tensor, &right_environment_tensor);
      return std::abs(tensor_to_scalar(contraction));
    }

    std::vector<ITensor> get_deltas_between(uint32_t q1, uint32_t q2) const {
      std::vector<ITensor> deltas;
      for (size_t q = q1; q < q2; q++) {
        Index idx = external_idx(q);
        deltas.push_back(delta(idx, prime(idx)));
      }

      return deltas;
    }

    double purity() {
      if (is_pure_state()) {
        return 1.0;
      }

      ITensor C = left_environment_tensor;
      ITensor L = C;
      L *= conj(prime(C, 2));
      for (size_t i = 0; i < num_qubits; i++) {
        C = partial_contraction(i, i+1, nullptr, nullptr, nullptr);
        L *= C;
        L *= conj(prime(prime(C, "Internal"), "Internal"));
      }
      L *= right_environment_tensor;
      L *= prime(right_environment_tensor, 2);
      return tensor_to_scalar(L).real();
    }

    size_t bond_dimension_at_site(size_t i) {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot check bond dimension of site {} for MPS with {} sites.", i, num_qubits));
      }

      orthogonalize(i);
      ITensor singular_values = svd_bond(i);

      return dim(inds(singular_values)[0]);
    }

    std::complex<double> inner(const MatrixProductStateImpl& other) const {
      if (!is_pure_state() || !other.is_pure_state()) {
        throw std::runtime_error("Can't compute inner product of mixed state.");
      }

      if (num_qubits != other.num_qubits) {
        throw std::runtime_error(fmt::format("Can't compute inner product of MPS; number of qubits do not match: {} and {}.", num_qubits, other.num_qubits));
      }
      
      Index ext1 = external_idx(0);
      Index ext2 = other.external_idx(0);
      ITensor contraction = tensors[0] * prime(conj(replaceInds(other.tensors[0], {ext2}, {ext1})), "Internal");

      for (size_t q = 1; q < num_qubits; q++) {
        ext1 = external_idx(q);
        ext2 = other.external_idx(q);
        contraction *= tensors[q];
        contraction *= prime(conj(replaceInds(other.tensors[q], {ext2}, {ext1})), "Internal");
      }

      contraction *= delta(left_boundary_index, prime(other.left_boundary_index));
      contraction *= delta(right_boundary_index, prime(other.right_boundary_index));
      return std::conj(tensor_to_scalar(contraction));
    }

    MatrixProductStateImpl concatenate(const MatrixProductStateImpl& other) const {
      if (dim(right_boundary_index) != 1 || dim(other.left_boundary_index) != 1) {
        throw std::runtime_error("Cannot concatenate MPS with nontrivial right (left) environment tensors.");
      }

      MatrixProductStateImpl new_impl(num_qubits + other.num_qubits, std::max(bond_dimension, other.bond_dimension), std::min(sv_threshold, other.sv_threshold));

      for (size_t i = 0; i < num_qubits; i++) {
        new_impl.tensors[i] = tensors[i];
        new_impl.external_indices[i] = external_indices[i];
      }

      for (size_t i = 0; i < other.num_qubits; i++) {
        uint32_t j = i + tensors.size();
        new_impl.tensors[j] = other.tensors[i];
      }

      for (size_t i = 0; i < internal_indices.size(); i++) {
        new_impl.internal_indices[i] = internal_indices[i];
      }

      // TODO FIX THIS
      Index internal = right_boundary_index;
      Index internal_ = Index(1, fmt::format("Internal,n={}",num_qubits));
      new_impl.tensors[num_qubits - 1].replaceInds({internal}, {internal_});

      internal = other.left_boundary_index;
      new_impl.tensors[num_qubits].replaceInds({internal}, {internal_});
      new_impl.internal_indices[num_qubits - 1] = internal_;

      // Correct internal indices
      for (size_t i = 0; i < other.internal_indices.size(); i++) {
        uint32_t j = i + internal_indices.size() + 1;
        Index internal = other.internal_indices[i];
        Index internal_ = Index(dim(internal), fmt::format("Internal,n={}",j));

        new_impl.tensors[j].replaceInds({internal}, {internal_});
        new_impl.tensors[j+1].replaceInds({internal}, {internal_});

        new_impl.internal_indices[j] = internal_;
      }

      for (size_t i = 0; i < other.num_qubits; i++) {
        uint32_t j = i + num_qubits;
        Index ext = findIndex(other.tensors[i], "External");
        Index ext_ = new_impl.external_indices[j];
        new_impl.tensors[j].replaceInds({ext}, {ext_});
      }

      new_impl.left_environment_tensor = left_environment_tensor;
      new_impl.right_environment_tensor = other.right_environment_tensor;

      new_impl.left_ortho_lim = 0;
      new_impl.right_ortho_lim = new_impl.num_qubits - 1;

      new_impl.left_boundary_index = left_boundary_index;
      new_impl.right_boundary_index = other.right_boundary_index;

      return new_impl;
    }

    void conjugate() {
      for (size_t i = 0; i < tensors.size(); i++) {
        orthogonalize(i);
        tensors[i] = conj(tensors[i]);
      }
      
      left_environment_tensor = conj(left_environment_tensor);
      right_environment_tensor = conj(right_environment_tensor);
    }

    void apply_measure(const MeasurementResult& result, const Qubits& qubits) {
      auto [q1, q2] = support_range(qubits).value();

      // TODO revisit this logic
      if (qubits.size() == 1) {
        const Eigen::Matrix2cd id = Eigen::Matrix2cd::Identity();
        if (q1 == 0) { // PI
          auto proj_ = Eigen::kroneckerProduct(id, result.proj);
          evolve(proj_, {q1, q1 + 1});
        } else { // IP
          auto proj_ = Eigen::kroneckerProduct(result.proj, id);
          evolve(proj_, {q1 - 1, q1});
        }
      } else {
        evolve(result.proj, qubits);
      }

      assert_state_valid(fmt::format("Error after applying measurement on {}.", qubits));
    }

    MeasurementResult measurement_result(const Measurement& m) {
      PauliString pauli = m.get_pauli();
      Qubits qubits = m.qubits;

      auto pm = pauli.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << pauli.num_qubits, 1u << pauli.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;
      Eigen::MatrixXcd proj1 = (id - pm)/2.0;

      double prob_zero = (1.0 + expectation(pauli.superstring(qubits, num_qubits)).real())/2.0;

      bool b;
      if (m.is_forced()) {
        b = m.get_outcome();
      } else {
        b = randf() > prob_zero;
      }

      QuantumState::check_forced_measure(b, prob_zero);
      auto proj = b ? proj1 : proj0;
      //auto proj = b ? proj1 / std::sqrt(1.0 - prob_zero) : proj0 / std::sqrt(prob_zero);

      return MeasurementResult(proj, prob_zero, b);
    }

    bool measure(const Measurement& m) {
      auto result = measurement_result(m);
      apply_measure(result, m.qubits);
      return result.outcome;
    }

    MeasurementResult weak_measurement_result(const WeakMeasurement& m) {
      PauliString pauli = m.get_pauli();
      Qubits qubits = m.qubits;

      auto pm = pauli.to_matrix();

      double prob_zero = (1 + std::tanh(2*m.beta) * expectation(pauli.superstring(qubits, num_qubits)).real())/2.0;

      bool b;
      if (m.is_forced()) {
        b = m.get_outcome();
      } else {
        b = randf() > prob_zero;
      }

      Eigen::MatrixXcd t = pm;
      if (b) {
        t = -t;
      }

      Eigen::MatrixXcd proj = (t*m.beta).exp();
      Eigen::MatrixXcd P = proj.pow(2);
      double norm = std::sqrt(std::abs(expectation(P, qubits)));

      proj = proj / norm;

      return MeasurementResult(proj, prob_zero, b);
    }

    bool weak_measure(const WeakMeasurement& m) {
      auto result = weak_measurement_result(m);
      apply_measure(result, m.qubits);
      return result.outcome;
    }

    // ======================================= DEBUG FUNCTIONS ======================================= //
    ITensor orthogonality_tensor_l(uint32_t q) const {
      ITensor A = tensors[q];
      Index right_index = right_boundary_index;
      if (q != num_qubits - 1) {
        right_index = internal_idx(q);
      }
      
      ITensor I = A * conj(prime(A, right_index));
      return I;
    }
    
    ITensor orthogonality_tensor_r(uint32_t q) const {
      ITensor A = tensors[q];
      Index left_index = left_boundary_index;
      if (q != 0) {
        left_index = internal_idx(q - 1);
      }

      ITensor I = A * conj(prime(A, left_index));
      return I;
    }

    std::string print_orthogonal_sites() const {
      std::stringstream stream;
      std::vector<double> ortho_l;
      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_l(i);
        ortho_l.push_back(distance_from_identity(I));
      }

      std::vector<double> ortho_r;
      for (size_t i = 0; i < num_qubits; i++) {
        auto I = orthogonality_tensor_r(i);
        ortho_r.push_back(distance_from_identity(I));
      }

      Qubits sites(num_qubits);
      std::iota(sites.begin(), sites.end(), 0);
      stream << fmt::format("ortho lims = {}, {}, tr = {:.4f}\n", left_ortho_lim, right_ortho_lim, trace());
      stream << fmt::format("          {::7}\n", sites);
      stream << fmt::format("ortho_l = {::.5f}\northo_r = {::.5f}\n", ortho_l, ortho_r);
      return stream.str();
    }

    bool check_orthonormality() const {
      if (orthogonality_level == 0) {
        return true;
      }

      for (size_t i = 0; i < left_ortho_lim; i++) {
        auto I = orthogonality_tensor_l(i);
        auto d = distance_from_identity(I);
        if (d > 1e-5) {
          return false;
        }
      }
      for (size_t i = right_ortho_lim + 1; i < num_qubits - 1; i++) {
        auto I = orthogonality_tensor_r(i);
        auto d = distance_from_identity(I);
        if (d > 1e-5) {
          return false;
        }
      }

      return true;
    }

    void state_checks(const std::string& error_message) const {
      // Orthonormal
      if (!check_orthonormality()) {
        throw std::runtime_error(fmt::format("{}\nError in orthogonality: \n{}\n", error_message, print_orthogonal_sites()));
      }
    }

    void assert_state_valid(const std::string& error_message = "") const {
      if (debug_level == 0) {
        return;
      } else if (debug_level == 1) {
        // Throw error but do not print state
        state_checks(error_message);
      } else {
        // Throw error and print state
        try {
          state_checks(error_message);
        } catch (const std::runtime_error& e) {
          print_mps();
          throw e;
        }
      }
    }
    // ======================================= DEBUG FUNCTIONS ======================================= //

    std::vector<char> to_bytes() const {
      std::vector<char> bytes;
      auto write_error = glz::write_beve(*this, bytes);
      if (write_error) {
        throw std::runtime_error(fmt::format("Error writing MatrixProductStateImpl to binary: \n{}", glz::format_error(write_error, bytes)));
      }
      return bytes;
    }

    void from_bytes(const std::vector<char>& bytes) {
      auto parse_error = glz::read_beve(*this, bytes);
      if (parse_error) {
        throw std::runtime_error(fmt::format("Error reading MatrixProductStateImpl from binary: \n{}", glz::format_error(parse_error, bytes)));
      }
    }
};


// ----------------------------------------------------------------------- //
// --------------- MatrixProductState implementation --------------------- //
// ----------------------------------------------------------------------- //

MatrixProductState::MatrixProductState(uint32_t num_qubits, uint32_t bond_dimension, double sv_threshold) : MagicQuantumState(num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(num_qubits, bond_dimension, sv_threshold);
}

MatrixProductState::MatrixProductState(const MatrixProductState& other) : MagicQuantumState(other.num_qubits) {
  impl = std::make_unique<MatrixProductStateImpl>(*other.impl.get());
}

MatrixProductState::MatrixProductState(const Statevector& other, uint32_t bond_dimension, double sv_threshold) : MatrixProductState(other.get_num_qubits(), bond_dimension, sv_threshold) {
  auto coefficients = vector_to_tensor(other.data, impl->external_indices);
  ITensor one_l(impl->left_boundary_index);
  one_l.set(1, 1.0);

  ITensor one_r(impl->right_boundary_index);
  one_r.set(1, 1.0);

  coefficients = coefficients * one_l * one_r;
  impl->reset_from_tensor(coefficients, std::make_pair(0, num_qubits));
}

MatrixProductState::MatrixProductState()=default;

MatrixProductState::~MatrixProductState()=default;

MatrixProductState& MatrixProductState::operator=(const MatrixProductState& other) {
  if (this != &other) {
    impl = std::make_unique<MatrixProductStateImpl>(*other.impl);
  }
  return *this;
}

MatrixProductState dmrg_state(SiteSet& sites, MPO& mpo, size_t num_sweeps, size_t num_qubits, size_t bond_dimension, double sv_threshold) {
  auto psi = randomMPS(sites, bond_dimension);

  auto sweeps = Sweeps(num_sweeps);
  sweeps.maxdim() = bond_dimension;
  sweeps.cutoff() = sv_threshold;
  sweeps.noise() = 1e-8;

  auto [energy, psi0] = dmrg(mpo, psi, sweeps, {"Silent=",true});
  psi0.normalize();

  auto impl = std::make_unique<MatrixProductStateImpl>(MatrixProductStateImpl::from_mps(psi0, bond_dimension, sv_threshold));
  impl->bond_dimension = bond_dimension;
  impl->sv_threshold = sv_threshold;

  impl->orthogonalize(0);

  MatrixProductState mps(num_qubits, bond_dimension, sv_threshold);
  mps.impl = std::move(impl);

  return mps;
}

MatrixProductState MatrixProductState::ising_ground_state(size_t num_qubits, double h, size_t max_bond_dimension, double sv_threshold, size_t num_sweeps) {
  SiteSet sites = SpinHalf(num_qubits, {"ConserveQNs=",false});

  auto ampo = AutoMPO(sites);
  for (int j = 1; j < num_qubits; ++j) {
    ampo += -2.0, "Sx", j, "Sx", j + 1;
  }

  for (int j = 1; j <= num_qubits; ++j) {
    ampo += -h, "Sz", j;
  }
  auto H = toMPO(ampo);

  return dmrg_state(sites, H, num_sweeps, num_qubits, max_bond_dimension, sv_threshold);
}

MatrixProductState MatrixProductState::xxz_ground_state(size_t num_qubits, double delta, size_t max_bond_dimension, double sv_threshold, size_t num_sweeps) {
  SiteSet sites = SpinHalf(num_qubits, {"ConserveQNs=",false});

  auto ampo = AutoMPO(sites);
  for (int j = 1; j < num_qubits; ++j) {
    ampo += -2.0,       "Sx", j, "Sx", j + 1;
    ampo += -2.0,       "Sy", j, "Sy", j + 1;
    ampo += -2.0*delta, "Sz", j, "Sz", j + 1;
  }
  auto H = toMPO(ampo);

  return dmrg_state(sites, H, num_sweeps, num_qubits, max_bond_dimension, sv_threshold);
}

std::string MatrixProductState::to_string() const {
  return impl->to_string();
}

double MatrixProductState::entanglement(const QubitSupport& support, uint32_t index) {
  // TODO add logic for intervals
  auto qubits = to_qubits(support);
	if (qubits.size() == 0) {
		return 0.0;
	}

	Qubits sorted_qubits(qubits);
	std::sort(sorted_qubits.begin(), sorted_qubits.end());

  // TODO add support for this with swap gates
  // beware exploding bond dimension
  if (!support_contiguous(sorted_qubits)) {
    throw std::runtime_error(fmt::format("Cannot compute MPS.entanglement of non-contiguous qubits {}.", qubits));
  }

  uint32_t first = sorted_qubits[0];
  uint32_t last = sorted_qubits[sorted_qubits.size() - 1];

  if (first == 0) {
    return impl->entanglement(last + 1, index);
  } else if (last == num_qubits - 1) {
    return impl->entanglement(first, index);
  } else {
    throw std::runtime_error(fmt::format("Cannot compute MPS.entanglement of qubits which do not border the boundary: {}", qubits));
  }
}

std::vector<double> MatrixProductState::singular_values(uint32_t i) const {
  return impl->singular_values_to_vector(i);
}

std::pair<std::vector<size_t>, std::vector<std::complex<double>>> MatrixProductState::tensor(uint32_t q) const {
  return impl->get_tensor(q);
}

double MatrixProductState::calculate_magic_mutual_information_from_samples2(const std::vector<double>& tAB, const std::vector<double>& tA, const std::vector<double>& tB) {
  if (tA.size() != tB.size() || tB.size() != tAB.size()) {
    throw std::invalid_argument(fmt::format("Invalid sample sizes passed to calculate_magic_from_samples2. tA.size() = {}, tB.size() = {}, tAB.size() = {}", tA.size(), tB.size(), tAB.size()));
  }

  size_t num_samples = tA.size();

  double I = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    I += std::pow(tA[i] * tB[i] / tAB[i], 2.0);
  }
  I = -std::log(I / num_samples);

  double W = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    W += std::pow(tA[i] * tB[i], 4.0) / std::pow(tAB[i], 2.0);
  }
  W = -std::log(W / num_samples);

  std::vector<double> tAB_(tAB);
  for (size_t i = 0; i < num_samples; i++) {
    tAB_[i] = tAB_[i] * tAB_[i];
  }

  double M = estimate_renyi_entropy(2, tAB_);
  return W - I - M;
}

std::vector<double> MatrixProductState::process_bipartite_pauli_samples(const std::vector<PauliAmplitudes>& samples) const {
  return impl->process_bipartite_pauli_samples(samples);
}

std::vector<BitAmplitudes> MatrixProductState::sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
  return impl->sample_bitstrings(supports, num_samples);
}

std::vector<std::vector<double>> MatrixProductState::process_bipartite_bit_samples(const std::vector<size_t>& renyi_indices, const std::vector<BitAmplitudes>& samples) const {
  return impl->process_bipartite_bit_samples(renyi_indices, samples);
}

std::vector<PauliAmplitudes> MatrixProductState::sample_paulis(const std::vector<QubitSupport>& supports, size_t num_samples) {
  return impl->sample_paulis(supports, num_samples);
}

std::complex<double> MatrixProductState::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

std::complex<double> MatrixProductState::expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const {
  return impl->expectation(m, qubits);
}

double MatrixProductState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  return impl->expectation(bits, support);
}

std::shared_ptr<QuantumState> MatrixProductState::partial_trace(const Qubits& qubits) const {
  auto interval = support_range(qubits);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > num_qubits) {
      throw std::runtime_error(fmt::format("qubits = {} passed to MatrixProductState.partial_trace with {} qubits", qubits, num_qubits));
    }
  }

  auto impl_ = std::make_unique<MatrixProductStateImpl>(impl->partial_trace(qubits));
  std::shared_ptr<MatrixProductState> mps = std::make_shared<MatrixProductState>(impl_->num_qubits, impl->bond_dimension, impl->sv_threshold);
  mps->impl = std::move(impl_);

  return mps;
}

MatrixProductState MatrixProductState::partial_trace_mps(const Qubits& qubits) const {
  auto interval = support_range(qubits);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > num_qubits) {
      throw std::runtime_error(fmt::format("qubits = {} passed to MatrixProductState.partial_trace with {} qubits", qubits, num_qubits));
    }
  }

  auto impl_ = std::make_unique<MatrixProductStateImpl>(impl->partial_trace(qubits));
  MatrixProductState mps(impl_->num_qubits, impl->bond_dimension, impl->sv_threshold);
  mps.impl = std::move(impl_);
  
  return mps;
}

bool MatrixProductState::is_pure_state() const {
  return impl->is_pure_state();
}

Eigen::MatrixXcd MatrixProductState::coefficients_mixed() const {
  return impl->coefficients_mixed();
}

Eigen::VectorXcd MatrixProductState::coefficients_pure() const {
  return impl->coefficients_pure();
}

double MatrixProductState::trace() const {
  // Do not assume normalization
  return impl->trace();
}

size_t MatrixProductState::bond_dimension(size_t i) const {
  return impl->bond_dimension_at_site(i);
}

void MatrixProductState::orthogonalize(uint32_t q) {
  impl->orthogonalize(q);
}

MatrixProductState MatrixProductState::concatenate(const MatrixProductState& other) const {
  MatrixProductState mps(num_qubits + other.num_qubits, std::max(impl->bond_dimension, other.impl->bond_dimension));
  auto new_impl = impl->concatenate(*other.impl.get());
  mps.impl = std::make_unique<MatrixProductStateImpl>(new_impl);
  return mps;
}

void MatrixProductState::conjugate() {
  impl->conjugate();
}

std::complex<double> MatrixProductState::inner(const MatrixProductState& other) const {
  return impl->inner(*other.impl.get());
}

void MatrixProductState::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
  impl->evolve(gate, qubit);
}

void MatrixProductState::evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  impl->evolve(gate, qubits);
}

std::vector<double> MatrixProductState::probabilities() const {
  if (impl->is_pure_state()) {
    Statevector psi(*this);
    return psi.probabilities();
  } else {
    DensityMatrix rho(*this);
    return rho.probabilities();
  }
}

double MatrixProductState::purity() const {
  return impl->purity();
}

bool MatrixProductState::measure(const Measurement& m) {
  return impl->measure(m);
}

bool MatrixProductState::weak_measure(const WeakMeasurement& m) {
  return impl->weak_measure(m);
}

std::vector<double> MatrixProductState::get_logged_truncerr() {
  std::vector<double> vals = impl->log;
  impl->log.clear();
  return vals;
}

// --- DEBUG FUNCTIONS
void MatrixProductState::print_mps(bool print_data) const {
  impl->print_mps(print_data);
}

void MatrixProductState::set_debug_level(int i) {
  impl->set_debug_level(i);
}

void MatrixProductState::set_orthogonality_level(int i) {
  impl->set_orthogonality_level(i);
}

bool MatrixProductState::state_valid() {
  try {
    impl->state_checks("Checking state externally.");
    return true;
  } catch (const std::runtime_error& error) {
    return false;
  }
}

struct MatrixProductState::glaze {
  using T = MatrixProductState;
  static constexpr auto value = glz::object(
    &T::impl,
    &T::use_parent,
    &T::num_qubits,
    &T::basis
  );
};

std::vector<char> MatrixProductState::serialize() const {
  std::vector<char> bytes;
  auto write_error = glz::write_beve(*this, bytes);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing MatrixProductState to binary: \n{}", glz::format_error(write_error, bytes)));
  }
  return bytes;
}

void MatrixProductState::deserialize(const std::vector<char>& bytes) {
  auto parse_error = glz::read_beve(*this, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error reading MatrixProductState from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}

void MatrixProductState::set_left_ortho_lim(uint32_t q) { impl->left_ortho_lim = q; }
void MatrixProductState::set_right_ortho_lim(uint32_t q) { impl->right_ortho_lim = q; }
