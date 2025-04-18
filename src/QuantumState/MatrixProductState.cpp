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

bool contiguous(const Qubits& v) {
  auto v_r = v;
  std::sort(v_r.begin(), v_r.end());

  for (size_t i = 0; i < v_r.size() - 1; i++) {
    if (v_r[i+1] != v_r[i] + 1) {
      return false;
    }
  }

  return true;
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
  friend class PauliExpectationTreeImpl;

  private:
		std::vector<ITensor> tensors;
		std::vector<ITensor> singular_values;
    std::vector<ITensor> blocks;

		std::vector<Index> external_indices;
		std::vector<Index> internal_indices;

    Index left_boundary_index;
    Index right_boundary_index;

    std::map<uint32_t, uint32_t> qubit_map;
    std::vector<uint32_t> qubit_indices;
    std::map<uint32_t, uint32_t> block_map;
    std::vector<uint32_t> block_indices;

    uint32_t left_ortho_lim;
    uint32_t right_ortho_lim;

    std::vector<double> log;
    int debug_level;
    int orthogonality_level;

    size_t num_blocks() const {
      return tensors.size() + blocks.size();
    }

  public:
    struct svd_error {
      unsigned int seed;
      std::optional<ITensor> T;
      std::vector<char> mps_bytes;
      uint32_t q;
      bool truncate;
    };

    static void write_svd_error(const std::string& filename, const svd_error& e) {
      std::vector<char> bytes;
      auto write_error = glz::write_beve(e, bytes);
      if (write_error) {
        throw std::runtime_error(fmt::format("Error writing svd_error to binary: \n{}", glz::format_error(write_error, bytes)));
      }

      std::ofstream output_file(filename, std::ios::out | std::ios::binary);
      if (!output_file) {
        throw std::runtime_error(fmt::format("Failed to open file for writing: {}", filename));
      }
      output_file.write(reinterpret_cast<const char*>(&bytes[0]), bytes.size());
      output_file.close();
    }

    static svd_error read_svd_error(const std::string& filename) {
      std::ifstream input_file(filename, std::ios::in | std::ios::binary);
      if (!input_file) {
        throw std::runtime_error(fmt::format("Failed to open file for reading: {}", filename));
      }

      std::vector<char> bytes((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
      input_file.close();

      svd_error e;
      auto parse_error = glz::read_beve(e, bytes);
      if (parse_error) {
        throw std::runtime_error(fmt::format("Error reading svd_error from binary: \n{}", glz::format_error(parse_error, bytes)));
      }

      return e;
    }

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
        &T::singular_values,
        &T::blocks,
        &T::internal_indices,
        &T::external_indices,
        &T::left_boundary_index,
        &T::right_boundary_index,
        &T::qubit_map,
        &T::qubit_indices,
        &T::block_map,
        &T::block_indices,
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
      : num_qubits(num_qubits), bond_dimension(bond_dimension), sv_threshold(sv_threshold), left_ortho_lim(num_qubits - 1), right_ortho_lim(0),
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

        blocks = {};
        block_indices = {};

        block_map = {};
        qubit_indices = std::vector<uint32_t>(num_qubits);
        std::iota(qubit_indices.begin(), qubit_indices.end(), 0);
        for (size_t i = 0; i < num_qubits; i++) {
          qubit_map[i] = i;
          qubit_indices[i] = i;
        }

        left_boundary_index = Index(1, "Internal,LEdge");
        right_boundary_index = Index(1, "Internal,REdge");

        for (uint32_t i = 0; i < num_qubits - 1; i++) {
          internal_indices.push_back(Index(1, fmt::format("Internal,Left,n={}", i)));
          internal_indices.push_back(Index(1, fmt::format("Internal,Right,n={}", i)));
        }

        for (uint32_t i = 0; i < num_qubits; i++) {
          qubit_map[i] = i;
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
        for (uint32_t q = 0; q < num_qubits - 1; q++) {
          tensor = ITensor(internal_idx(q, InternalDir::Left), internal_idx(q, InternalDir::Right));
          tensor.set(1, 1, 1.0);
          singular_values.push_back(tensor);
        }

        // Setting left boundary tensor
        tensor = ITensor(left_boundary_index, internal_idx(0, InternalDir::Left), external_idx(0));
        tensor.set(1, 1, 1, 1.0);
        tensors.push_back(tensor);

        // Setting bulk tensors
        for (uint32_t q = 1; q < num_qubits - 1; q++) {
          tensor = ITensor(internal_idx(q - 1, InternalDir::Right), internal_idx(q, InternalDir::Left), external_idx(q));
          tensor.set(1, 1, 1, 1.0);
          tensors.push_back(tensor);
        }

        // Setting right boundary tensor
        tensor = ITensor(internal_idx(num_qubits - 2, InternalDir::Right), right_boundary_index, external_idx(num_qubits - 1));
        tensor.set(1, 1, 1, 1.0);
        tensors.push_back(tensor);
      }

    MatrixProductStateImpl(const MatrixProductStateImpl& other) : MatrixProductStateImpl(other.num_qubits, other.bond_dimension, other.sv_threshold) {
      tensors = other.tensors;
      singular_values = other.singular_values;
      blocks = other.blocks;

      qubit_map = other.qubit_map;
      qubit_indices = other.qubit_indices;
      block_map = other.block_map;
      block_indices = other.block_indices;

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

    MatrixProductStateImpl partial_trace(const Qubits& qubits) {
      orthogonalize(0);

      std::set<uint32_t> traced(qubits.begin(), qubits.end());

      std::vector<ITensor> tensors_;
      std::vector<ITensor> singular_values_;
      std::vector<ITensor> blocks_;

      std::map<uint32_t, uint32_t> qubit_map_;
      std::vector<uint32_t> qubit_indices_;
      std::map<uint32_t, uint32_t> block_map_;
      std::vector<uint32_t> block_indices_;

      std::vector<Index> internal_indices_;
      std::vector<Index> external_indices_;

      size_t k = 0;
      size_t i = 0;
      size_t qubit_i = 0;
      size_t block_i = 0;


      auto get_block_starting_at_index = [&](size_t& k) {
        size_t i1 = k;
        size_t i2 = k;
        bool keep_going = true;
        while (keep_going && i2 < num_blocks()) {
          i2++;
          if (qubit_map.contains(i2)) {
            size_t q_ = qubit_map.at(i2);
            if (!traced.contains(q_)) {
              keep_going = false;
            }
          }
        }

        std::vector<ITensor> external_tensors = get_deltas_between(i1, i2);
        InternalDir direction = (i2 == num_blocks()) ? InternalDir::Right : InternalDir::Left;
        auto block = partial_contraction(i1, i2, &external_tensors, nullptr, nullptr, false, direction);
        if (order(block) > 4) {
          throw std::runtime_error("Something went wrong with partial_contraction of blocks.");
        }

        k = i2;
        return block;
      };

      while (k < num_blocks()) {
        if (qubit_map.contains(k)) {
          size_t q = qubit_map.at(k);
          if (traced.contains(q)) {
            auto block = get_block_starting_at_index(k);
            blocks_.push_back(block);
            block_map_[i] = block_i++;
            block_indices_.push_back(i);
          } else {
            ITensor tensor = tensors[q];
            tensors_.push_back(tensor);
            qubit_map_[i] = qubit_i++;
            qubit_indices_.push_back(i);
            k++;
          }
        } else {
          size_t q = block_map.at(k);
          blocks_.push_back(blocks[q]);
          block_map_[i] = block_i++;
          block_indices_.push_back(i);
          k++;
        }

        if (k != num_blocks()) {
          singular_values_.push_back(singular_values[k - 1]);
        }

        i++;
      }

      for (size_t j = 0; j < i - 1; j++) {
        auto left = findIndex(singular_values_[j], "Left");
        auto right = findIndex(singular_values_[j], "Right");

        Index left_(dim(left), fmt::format("n={},Internal,Left", j));
        Index right_(dim(right), fmt::format("n={},Internal,Right", j));

        try { 
          singular_values_[j].replaceInds({left, right}, {left_, right_});
        } catch (const ITError& e) {
          singular_values_[j] = toDense(singular_values_[j]);
          singular_values_[j].replaceInds({left, right}, {left_, right_});
        }
        if (qubit_map_.contains(j)) {
          size_t q = qubit_map_.at(j);
          tensors_[q].replaceInds({left}, {left_});
        } else {
          size_t b = block_map_.at(j);
          blocks_[b].replaceInds({left, prime(left)}, {left_, prime(left_)});
        }

        if (qubit_map_.contains(j + 1)) {
          size_t q = qubit_map_.at(j + 1);
          tensors_[q].replaceInds({right}, {right_});
        } else {
          size_t b = block_map_.at(j + 1);
          blocks_[b].replaceInds({right, prime(right)}, {right_, prime(right_)});
        }
        
        internal_indices_.push_back(left_);
        internal_indices_.push_back(right_);
      }


      size_t remaining_qubits = num_qubits - qubits.size();
      for (size_t q = 0; q < remaining_qubits; q++) {
        Index external = findIndex(tensors_[q], "External");
        Index external_(dim(external), fmt::format("i={},External", q));
        tensors_[q].replaceInds({external}, {external_});
        external_indices_.push_back(external_);
      }

      MatrixProductStateImpl mps(remaining_qubits, bond_dimension, sv_threshold);
      
      mps.tensors = tensors_;
      mps.blocks = blocks_;
      mps.singular_values = singular_values_;

      mps.qubit_map = qubit_map_;
      mps.qubit_indices = qubit_indices_;
      mps.block_map = block_map_;
      mps.block_indices = block_indices_;

      mps.left_boundary_index = left_boundary_index;
      mps.right_boundary_index = right_boundary_index;

      mps.external_indices = external_indices_;
      mps.internal_indices = internal_indices_;

      // TODO Need to find a better way to do this than just turning off orthogonalization
      mps.left_ortho_lim = 0;
      mps.right_ortho_lim = 0;
      mps.debug_level = debug_level;
      mps.orthogonality_level = 0; 

      return mps;
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
      if (i >= internal_indices.size()) {
        throw std::runtime_error(fmt::format("Cannot retrieve internal index for i = {}.", i));
      }

      if (d == InternalDir::Left) {
        return internal_indices[2*i];
      } else {
        return internal_indices[2*i + 1];
      }
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
      auto print_tensor_at_index = [&](size_t i) {
        if (qubit_map.contains(i)) {
          size_t q = qubit_map.at(i);
          std::cout << fmt::format("Uncontracted qubit at {}\n", i);
          if (print_data) {
            PrintData(tensors[q]);
          } else {
            print(tensors[q]);
          }
        } else {
          size_t b = block_map.at(i);
          std::cout << fmt::format("Contracted block at {}\n", i);
          if (print_data) {
            PrintData(blocks[b]);
          } else {
            print(blocks[b]);
          }
        }
      };

      print_tensor_at_index(0);

      for (size_t i = 0; i < num_blocks() - 1; i++) {
        if (print_data) {
          PrintData(singular_values[i]);
        } else {
          print(singular_values[i]);
        }

        print_tensor_at_index(i + 1);
      }
    }

    void set_left_ortho_lim(uint32_t q) {
      left_ortho_lim = std::min(left_ortho_lim, q);
    }

    void set_right_ortho_lim(uint32_t q) {
      right_ortho_lim = std::max(right_ortho_lim, q);
    }

    void left_orthogonalize(uint32_t q) {
      while (left_ortho_lim < q) {
        svd_bond(left_ortho_lim++, nullptr, false);
        if (left_ortho_lim > right_ortho_lim) {
          right_ortho_lim++;
        }
      }
    }

    void right_orthogonalize(uint32_t q) {
      while (right_ortho_lim > q) {
        svd_bond(--right_ortho_lim, nullptr, false);
        if (right_ortho_lim < left_ortho_lim) {
          left_ortho_lim--;
        }
      }
    }

    void orthogonalize(size_t q) {
      if (orthogonality_level == 0) {
        return;
      }

      if (q > num_qubits - 1 || q < 0) {
        throw std::runtime_error(fmt::format("Cannot move orthogonality center of state with {} qubits to site {}\n", num_qubits, q));
      }

      left_orthogonalize(q);
      right_orthogonalize(q);
    }

    void orthogonalize(size_t i1, size_t i2) {
      if (orthogonality_level == 0) {
        return;
      }

      size_t j1 = i1;
      while (!qubit_map.contains(j1) && j1 < num_blocks()) {
        j1++;
      }

      size_t q1 = qubit_map.at(j1);
      left_orthogonalize(q1);

      size_t j2 = i2;
      while (!qubit_map.contains(j2) && j2 > 0) {
        j2--;
      }

      size_t q2 = qubit_map.at(j2);
      right_orthogonalize(q2 - 1);
    }

    bool is_orthogonal() const {
      if (orthogonality_level == 0) {
        return true;
      } else {
        return left_ortho_lim >= right_ortho_lim;
      }
    }

    std::vector<double> singular_values_to_vector(size_t i) const {
      if (i >= singular_values.size()) {
        throw std::runtime_error(fmt::format("Cannot retrieve singular values in index {} for MPS with {} blocks.", i, num_blocks()));
      }

      std::vector<double> sv(bond_dimension, 0.0);
      uint32_t N = dim(inds(singular_values[i])[0]);
      for (uint32_t j = 0; j < N; j++) {
        std::vector<uint32_t> assignments{j + 1, j + 1};
        sv[j] = elt(singular_values[i], assignments);
      }

      return sv;
    }

    std::vector<std::vector<std::vector<std::complex<double>>>> get_tensor(uint32_t q) const {
      ITensor T = tensors[q];

      Index ext = external_idx(q);

      Index left_idx;
      if (q == 0) {
        left_idx = left_boundary_index;
      } else {
        T *= singular_values[q - 1];
        left_idx = internal_idx(q - 1, InternalDir::Left);
      }
      size_t d1 = dim(left_idx);

      Index right_idx;
      if (q == num_qubits - 1) {
        right_idx = right_boundary_index;
      } else {
        right_idx = internal_idx(q, InternalDir::Left);
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

    double entropy(uint32_t q, uint32_t index) {
      if (q < 0 || q > num_qubits) {
        throw std::invalid_argument("Invalid qubit passed to MatrixProductState.entropy; must have 0 <= q <= num_qubits.");
      }

      if (q == 0 || q == num_qubits) {
        return 0.0;
      }

      orthogonalize(q);
      assert_state_valid(fmt::format("State invalid after applying entropy({})\n", q));

      auto singular_vals = singular_values[q-1];
      int d = dim(inds(singular_vals)[0]);

      std::vector<double> sv(d);
      for (int i = 0; i < d; i++) {
        sv[i] = std::pow(elt(singular_vals, i+1, i+1), 2);
      }

      double s = 0.0;
      if (index == 1) {
        for (double v : sv) {
          if (v >= 1e-6) {
            s -= v * std::log(v);
          }
        }
      } else {
        for (double v : sv) {
          s += std::pow(v, index);
        }
        
        s = std::log(s)/(1.0 - index);
      }

      return s;
    }

    ITensor left_boundary_tensor(size_t i) const {
      Index left_idx;
      if (i == 0) {
        left_idx = left_boundary_index;
      } else {
        left_idx = internal_idx(i-1, InternalDir::Left);
      }
      return delta(left_idx, prime(left_idx));
    }

    ITensor left_environment_tensor(size_t i, const std::vector<ITensor>& external_tensors) const {
      ITensor L = left_boundary_tensor(0);
      extend_left_environment_tensor(L, 0, i, external_tensors);
      return L;
    }

    ITensor left_environment_tensor(size_t i) const {
      std::vector<ITensor> external_tensors = get_deltas_between(0, i);
      return left_environment_tensor(i, external_tensors);
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
      if (i == num_blocks()) {
        return delta(right_boundary_index, prime(right_boundary_index));
      } else {
        Index right_idx = internal_idx(i - 1, InternalDir::Left);
        ITensor tensor;
        try {
          tensor = singular_values[i - 1] * conj(prime(singular_values[i - 1], "Left"));
        } catch (const ITError& e) {
          tensor = toDense(singular_values[i - 1]) * conj(prime(singular_values[i - 1], "Left"));
        }
        Index right_idx_ = noPrime(inds(tensor)[0]);

        return replaceInds(tensor, {right_idx_, prime(right_idx_)}, {right_idx, prime(right_idx)});
      }
    }

    ITensor right_environment_tensor(size_t i, const std::vector<ITensor>& external_tensors) const {
      ITensor R = right_boundary_tensor(num_blocks());
      extend_right_environment_tensor(R, num_blocks(), i, external_tensors);
      return R;
    }

    ITensor right_environment_tensor(size_t i) const {
      std::vector<ITensor> external_tensors = get_deltas_between(i, num_blocks());
      return right_environment_tensor(i, external_tensors);
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t i1, uint32_t i2, const std::vector<ITensor>& external_tensors) const {
      if (i1 != i2) {
        R = partial_contraction(i2, i1, &external_tensors, nullptr, &R, true, InternalDir::Right);
      }
    }

    void extend_right_environment_tensor(ITensor& R, uint32_t i1, uint32_t i2) const {
      std::vector<ITensor> external_tensors = get_deltas_between(i2, i1);
      extend_right_environment_tensor(R, i1, i2, external_tensors);
    }

    ITensor right_tensor(size_t i) const {
      if (orthogonality_level == 0) {
        return right_environment_tensor(i);
      } else {
        return right_boundary_tensor(i);
      }
    }

    ITensor partial_contraction(size_t i1, size_t i2, const std::vector<ITensor>* external_tensors, const ITensor* L, const ITensor* R, bool include_sv=true, InternalDir direction=InternalDir::Left) const {
      bool left = (direction == InternalDir::Left);

      ITensor contraction;

      size_t k = 0;

      auto add_tensor = [](ITensor& C, ITensor& tensor) {
        if (!C) {
          C = tensor;
        } else {
          C *= tensor;
        }
      };

      auto advance_contraction = [&](ITensor& C, size_t j, bool include_sv) {
        if (qubit_map.contains(j)) {
          size_t q = qubit_map.at(j);
          ITensor tensor = tensors[q];

          if (j != 0 && include_sv) {
            tensor *= singular_values[j - 1];
          }

          add_tensor(C, tensor);

          if (external_tensors != nullptr && k < external_tensors->size()) {
            size_t p = left ? k : (external_tensors->size() - 1 - k);
            k++;
            C *= (*external_tensors)[p];
          }
          C *= conj(prime(tensor));
        } else {
          size_t b = block_map.at(j);
          ITensor block = blocks[b];

          if (j != 0 && include_sv) {
            block *= singular_values[j - 1];
            block *= prime(singular_values[j - 1]);
          }

          add_tensor(C, block);
        }
      };

      const ITensor* T1 = left ? L : R;
      const ITensor* T2 = left ? R : L;

      if (T1 != nullptr) {
        contraction = *T1;
      }

      size_t width = i2 - i1;

      for (size_t j = 0; j < width; j++) {
        size_t n = left ? (i1 + j) : (i2 - j - 1);
        bool include_sv_ = true;
        if (n == i1) {
          include_sv_ = include_sv;
        }

        advance_contraction(contraction, n, include_sv_);
      }

      if (T2 != nullptr) {
        contraction *= (*T2);
      }

      return contraction;
    }

    std::complex<double> partial_expectation(const Eigen::MatrixXcd& m, uint32_t i1, uint32_t i2, const ITensor& L, const ITensor& R) const {
      std::vector<Index> idxs;
      std::vector<Index> idxs_;

      for (size_t i = i1; i < i2; i++) {
        if (qubit_map.contains(i)) {
          size_t q = qubit_map.at(i);

          Index idx = external_idx(q);
          idxs.push_back(idx);
          idxs_.push_back(prime(idx));
        }
      }

      std::vector<ITensor> mtensor = {matrix_to_tensor(m, idxs_, idxs)};
      ITensor contraction = partial_contraction(i1, i2, &mtensor, &L, &R);
      return tensor_to_scalar(contraction);
    }

    std::complex<double> partial_expectation(const std::vector<ITensor>& operators, uint32_t i1, uint32_t i2, const ITensor& L, const ITensor& R) const {
      ITensor contraction = partial_contraction(i1, i2, &operators, &L, &R);
      return tensor_to_scalar(contraction);
    }

    std::pair<ITensor, ITensor> get_boundary_tensors(size_t i1, size_t i2) {
      if (orthogonality_level == 0) {
        return {left_environment_tensor(i1), right_environment_tensor(i2)};
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

      size_t i1 = qubit_indices[q1];
      size_t i2 = qubit_indices[q2 - 1] + 1;

      auto [L, R] = get_boundary_tensors(i1, i2);

      std::vector<ITensor> paulis;
      for (size_t i = i1; i < i2; i++) {
        if (qubit_map.contains(i)) {
          size_t q = qubit_map.at(i);
          Index idx = external_idx(q);
          paulis.push_back(pauli_tensor(p.to_pauli(q), prime(idx), idx));
        }
      }

      return p.sign() * partial_expectation(paulis, i1, i2, L, R);
    }

    // TODO make this provide marginal probabilities when Qubits are passed
    double expectation(const BitString& bits, std::optional<QubitSupport>& support) {
      size_t i1, i2;
      size_t q1, q2;
      if (support) {
        QubitInterval interval = to_interval(support.value());
        if (interval) {
          std::tie(q1, q2) = interval.value();
        } else {
          return 1.0;
        }
      } else {
        q1 = 0;
        q2 = num_qubits;
      }

      size_t i1 = qubit_indices[q1];
      size_t i2 = qubit_indices[q2 - 1] + 1;

      auto [L, R] = get_boundary_tensors(i1, i2);

      std::vector<ITensor> projectors;
      Eigen::Matrix2cd P1; P1 << 0, 0, 0, 1;
      Eigen::Matrix2cd P0; P0 << 1, 0, 0, 0;
      for (size_t i = i1; i < i2; i++) {
        if (qubit_map.contains(i)) {
          size_t q = qubit_map.at(i);
          Index idx = external_idx(q);
          if (q >
          if (bits.get(q)) {
            projectors.push_back(matrix_to_tensor(P1, {prime(idx)}, {idx}));
          } else {
            projectors.push_back(matrix_to_tensor(P0, {prime(idx)}, {idx}));
          }
        }
      }

      return partial_expectation(projectors, i1, i2, L, R).real();
    }

    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) {
      size_t r = m.rows();
      size_t c = m.cols();
      size_t n = qubits.size();
      if (r != c || (1u << n != r)) {
        throw std::runtime_error(fmt::format("Passed observable has dimension {}x{}, provided {} sites.", r, c, n));
      }

      if (!contiguous(qubits)) {
        throw std::runtime_error(fmt::format("Provided sites {} are not contiguous.", qubits));
      }

      auto [q1, q2] = to_interval(qubits).value();

      size_t i1 = qubit_indices[q1];
      size_t i2 = qubit_indices[q2 - 1] + 1;
      
      auto [L, R] = get_boundary_tensors(i1, i2);

      return partial_expectation(m, i1, i2, L, R);
    }

    std::vector<BitAmplitudes> sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) {
      int i = orthogonality_level;
      set_orthogonality_level(1);
      orthogonalize(0);
      set_orthogonality_level(i);

      std::vector<BitAmplitudes> samples;
      
      Eigen::Matrix2cd P0; P0 << 1.0, 0.0, 0.0, 0.0;
      Eigen::Matrix2cd P1; P1 << 0.0, 0.0, 0.0, 1.0;
      for (size_t i = 0; i < num_samples; i++) {
        double p = 1.0;
        BitString bits(num_qubits);
        ITensor L = left_boundary_tensor(qubit_indices[0]);

        for (size_t q = 0; q < num_qubits; q++) {
          size_t j1 = qubit_indices[q];
          size_t j2 = index_of_next_qubit(q);

          Index ext = external_idx(q);
          std::vector<ITensor> M = {matrix_to_tensor(P0, {prime(ext)}, {ext})};

          ITensor R = right_boundary_tensor(j2);
          ITensor L0 = partial_contraction(j1, j2, &M, &L, nullptr);

          double p0 = std::abs(tensor_to_scalar(L0 * R))/p;
          
          bool v = (randf() >= p0);
          bits.set(q, v);
          if (v) {
            M = {matrix_to_tensor(P1, {prime(ext)}, {ext})};
            ITensor L1 = partial_contraction(j1, j2, &M, &L, nullptr);
            L = L1;
            p *= 1.0 - p0;
          } else {
            L = L0;
            p *= p0;
          }
        }

        // TODO support for supports
        samples.push_back({bits, {p}});
      }

      return samples;
    }

    size_t index_of_next_qubit(size_t q) const {
      if (q == num_qubits - 1) {
        return num_blocks();
      } else {
        return qubit_indices[q + 1];
      }
    }

    PauliAmplitudes sample_pauli(const std::vector<QubitSupport>& supports, std::minstd_rand& rng) {
      int i = orthogonality_level;
      set_orthogonality_level(1);
      orthogonalize(0);
      set_orthogonality_level(i);

      std::vector<Pauli> p(num_qubits);
      double P = 1.0;

      Index left = left_boundary_index;
      ITensor L = left_boundary_tensor(qubit_indices[0]);

      for (size_t q = 0; q < num_qubits; q++) {
        size_t j1 = qubit_indices[q];
        size_t j2 = index_of_next_qubit(q);

        std::vector<double> probs(4);
        std::vector<ITensor> pauli_tensors(4);

        Index s = external_indices[q];

        for (size_t p = 0; p < 4; p++) {
          std::vector<ITensor> sigma = {pauli_tensor(static_cast<Pauli>(p), prime(s), s)};
          auto C = partial_contraction(j1, j2, &sigma, &L, nullptr, false);
          if (q != num_blocks() - 1) {
            C *= singular_values[q];
            C *= prime(singular_values[q]);
          }

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

      return {pauli, amplitudes};
    }

    std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& supports, size_t num_samples) {
      std::vector<PauliAmplitudes> samples(num_samples);

      std::minstd_rand rng(randi());
      for (size_t k = 0; k < num_samples; k++) {
        samples[k] = sample_pauli(supports, rng);
      } 

      return samples;
    }

    std::vector<PauliAmplitudes> sample_paulis_montecarlo(
      PauliExpectationTree& tree, const std::vector<QubitSupport>& supports, 
      size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, PauliMutationFunc mutation
    ) {
      PauliString p = tree.to_pauli_string();
      auto perform_mutation = [&](PauliString& p) -> double {
        double t1 = std::abs(tree.expectation());
        double p1 = prob(t1);

        PauliString q(p);
        mutation(q);

        PauliString product = q*p;

        tree.modify(product);

        double t2 = std::abs(tree.expectation());
        double p2 = prob(t2);

        double r = randf();
        if (r < p2 / p1) {
          p = PauliString(q);
          return t2;
        } else {
          // Put state back
          tree.modify(product);
          return t1;
        }
      };

      for (size_t i = 0; i < equilibration_timesteps; i++) {
        double t = perform_mutation(p);
      }

      std::vector<PauliAmplitudes> samples(num_samples);
      for (size_t i = 0; i < num_samples; i++) {
        double t = perform_mutation(p);
        std::vector<double> amplitudes{t};
        for (const auto& support : supports) {
          PauliString ps = p.substring(support, false);
          auto interval = ps.support_range();
          if (interval) {
            auto [q1, q2] = interval.value();
            double ts = std::abs(tree.partial_expectation(q1, q2));
            amplitudes.push_back(ts);
          } else {
            amplitudes.push_back(1.0);
          }
        }
        samples[i] = {p, amplitudes};
      }

      return samples;
    }

    std::vector<double> process_bipartite_pauli_samples(const std::vector<PauliAmplitudes>& pauli_samples) {
      int i = orthogonality_level;
      set_orthogonality_level(1);
      orthogonalize(0);

      size_t N = num_qubits/2 - 1;
      size_t num_samples = pauli_samples.size();
      std::vector<std::vector<double>> samplesA(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesB(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesAB(N, std::vector<double>(num_samples));

      for (size_t j = 0; j < num_samples; j++) {
        auto const [P, t] = pauli_samples[j];

        std::vector<double> tA(N);
        ITensor L = left_boundary_tensor(qubit_indices[0]);
        for (size_t q = 0; q < N; q++) {
          Index idx = external_idx(q);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q), prime(idx), idx)};
          uint32_t i1 = qubit_indices[q];
          uint32_t i2 = index_of_next_qubit(q);
          extend_left_environment_tensor(L, i1, i2, p);

          ITensor contraction = L * right_boundary_tensor(i2);
          samplesA[q][j] = std::abs(tensor_to_scalar(contraction));
        }

        std::vector<double> tB(N);
        ITensor R = right_boundary_tensor(num_blocks());
        std::vector<ITensor> paulis;
        for (size_t q = num_qubits/2; q < num_qubits; q++) {
          Index idx = external_idx(q);
          paulis.push_back(pauli_tensor(P.to_pauli(q), prime(idx), idx));
        }
        size_t i = qubit_indices[num_qubits/2];
        extend_right_environment_tensor(R, num_blocks(), i, paulis);

        for (size_t n = 0; n < N; n++) {
          uint32_t q = num_qubits/2 - n;
          Index idx = external_idx(q - 1);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q - 1), prime(idx), idx)};
          uint32_t i1 = qubit_indices[q];
          uint32_t i2 = qubit_indices[q - 1];
          extend_right_environment_tensor(R, i1, i2, p);

          ITensor contraction = left_boundary_tensor(i2) * R;
          samplesB[N - 1 - n][j] = std::abs(tensor_to_scalar(contraction));
        }

        for (size_t n = 0; n < N; n++) {
          samplesAB[n][j] = t[0];
        }
      }

      std::vector<double> magic(N);
      for (size_t n = 0; n < N; n++) {
        magic[n] = MagicQuantumState::calculate_magic_mutual_information_from_samples2({samplesAB[n], samplesA[n], samplesB[n]});
      }

      set_orthogonality_level(i);
      return magic;
    }

    std::vector<double> process_bipartite_bit_samples(const std::vector<BitAmplitudes>& bit_samples) {
      int i = orthogonality_level;
      set_orthogonality_level(1);
      orthogonalize(0);

      size_t N = num_qubits/2 - 1;
      size_t num_samples = bit_samples.size();
      std::vector<std::vector<double>> samplesA(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesB(N, std::vector<double>(num_samples));
      std::vector<std::vector<double>> samplesAB(N, std::vector<double>(num_samples));

      for (size_t j = 0; j < num_samples; j++) {
        auto const [b, t] = bit_samples[j];

        std::vector<double> tA(N);
        ITensor L = left_boundary_tensor(qubit_indices[0]);
        for (size_t q = 0; q < N; q++) {
          Index idx = external_idx(q);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q), prime(idx), idx)};
          uint32_t i1 = qubit_indices[q];
          uint32_t i2 = index_of_next_qubit(q);
          extend_left_environment_tensor(L, i1, i2, p);

          ITensor contraction = L * right_boundary_tensor(i2);
          samplesA[q][j] = std::abs(tensor_to_scalar(contraction));
        }

        std::vector<double> tB(N);
        ITensor R = right_boundary_tensor(num_blocks());
        std::vector<ITensor> paulis;
        for (size_t q = num_qubits/2; q < num_qubits; q++) {
          Index idx = external_idx(q);
          paulis.push_back(pauli_tensor(P.to_pauli(q), prime(idx), idx));
        }
        size_t i = qubit_indices[num_qubits/2];
        extend_right_environment_tensor(R, num_blocks(), i, paulis);

        for (size_t n = 0; n < N; n++) {
          uint32_t q = num_qubits/2 - n;
          Index idx = external_idx(q - 1);
          std::vector<ITensor> p = {pauli_tensor(P.to_pauli(q - 1), prime(idx), idx)};
          uint32_t i1 = qubit_indices[q];
          uint32_t i2 = qubit_indices[q - 1];
          extend_right_environment_tensor(R, i1, i2, p);

          ITensor contraction = left_boundary_tensor(i2) * R;
          samplesB[N - 1 - n][j] = std::abs(tensor_to_scalar(contraction));
        }

        for (size_t n = 0; n < N; n++) {
          samplesAB[n][j] = t[0];
        }
      }

      std::vector<double> magic(N);
      for (size_t n = 0; n < N; n++) {
        magic[n] = MagicQuantumState::calculate_magic_mutual_information_from_samples2({samplesAB[n], samplesA[n], samplesB[n]});
      }

      set_orthogonality_level(i);
      return magic;
    }

    bool is_pure_state() const {
      return blocks.size() == 0;
    }

    Eigen::MatrixXcd coefficients_mixed() {
      if (num_qubits > 15) {
        throw std::runtime_error("Cannot generate coefficients for n > 15 qubits.");
      }

      ITensor L = left_boundary_tensor(0);
      ITensor R = right_boundary_tensor(num_blocks());
      auto contraction = partial_contraction(0, num_blocks(), nullptr, &L, &R);

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

      return data.conjugate();
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
        C *= singular_values[q]*tensors[q+1];
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

    void svd_bond(uint32_t q, ITensor* T=nullptr, bool truncate=true) {
      size_t j1 = qubit_indices[q];
      size_t j2 = qubit_indices[q+1];

      if (j2 - j1 > 1) {
        throw std::runtime_error("Cannot currently perform svd_bond across qubits which have been partially traced.");
      }

      ITensor theta = tensors[q];
      theta *= singular_values[j1];
      theta *= tensors[q+1];
      if (T) {
        theta = noPrime(theta * (*T));
      }

      std::vector<Index> u_inds{external_idx(q)};
      std::vector<Index> v_inds{external_idx(q+1)};

      Index left;
      if (j1 != 0) {
        left = internal_idx(j1 - 1, InternalDir::Left);
        theta *= singular_values[j1 - 1];
      } else {
        left = left_boundary_index;
      }
      u_inds.push_back(left);

      Index right;
      if (j2 != num_blocks() - 1) {
        right = internal_idx(j2, InternalDir::Right);
        theta *= singular_values[j2];
      } else {
        right = right_boundary_index;
      }
      v_inds.push_back(right);

      double threshold = truncate ? sv_threshold : 1e-15;

      ITensor U, S, V;
      try {
        theta = apply(theta, [](Cplx c) { 
          double mask = std::abs(c) >= 1e-14;
          return Cplx(c.real() * mask, c.imag() * mask);
        });

        std::tie(U, S, V) = svd(theta, u_inds, v_inds, 
            {"Cutoff=",threshold,"MaxDim=",bond_dimension,
             "LeftTags=",fmt::format("Internal,Left,n={}", q),
             "RightTags=",fmt::format("Internal,Right,n={}", q)});
      } catch (const std::runtime_error& e) {
        std::optional<ITensor> t = std::nullopt;
        if (T) {
          t = *T;
        }
        svd_error error{Random::get_seed(), t, to_bytes(), q, truncate};
        uint32_t r = randi();
        write_svd_error(fmt::format("svd_error{:05}.eve", r), error);
        std::cout << "There was a LAPACK error!\n";
        writeToFile("theta.h5", theta);
        throw e;
      }

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

      auto inv = [&](Real r) { 
        if (r > sv_threshold) {
          return 1.0/r;
        } else {
          return 0.0;
        }
      };

      if (j1 != 0) {
        U *= apply(singular_values[j1 - 1], inv);
      }
      if (j2 != num_blocks() - 1) {
        V *= apply(singular_values[j2], inv);
      }

      internal_indices[2*j1] = commonIndex(U, S);
      internal_indices[2*j1 + 1] = commonIndex(V, S);

      tensors[q] = U;
      tensors[q+1] = V;
      singular_values[j1] = S;
    }

    void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
      auto i = external_indices[qubit];
      auto ip = prime(i);
      ITensor tensor = matrix_to_tensor(gate, {ip}, {i});
      tensors[qubit] = noPrime(tensors[qubit]*tensor);

      std::stringstream stream;
      stream << "Error after applying gate \n" << gate << fmt::format("\n to qubit {}.", qubit);
      assert_state_valid(stream.str());
    }

    void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
      assert_gate_shape(gate, qubits);

      if (qubits.size() == 1) {
        evolve(gate, qubits[0]);
        return;
      }

      if ((qubits.size()) != 2 || (gate.rows() != gate.cols()) || (gate.rows() != (1u << qubits.size()))) {
        throw std::invalid_argument("Can only evolve two-qubit gates in MPS simulation.");
      }

      uint32_t q1 = std::min(qubits[0], qubits[1]);
      uint32_t q2 = std::max(qubits[0], qubits[1]);

      orthogonalize(q1);

      if (q2 - q1 > 1) {
        for (size_t q = q1; q < q2 - 1; q++) {
          swap(q, q+1);
        }

        Qubits qbits{q2 - 1, q2};
        if (qubits[0] > qubits[1]) {
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

      auto i1 = external_idx(qubits[0]);
      auto i2 = external_idx(qubits[1]);
      ITensor gate_tensor = matrix_to_tensor(gate, 
        {prime(i1), prime(i2)}, 
        {i1, i2}
      );

      svd_bond(q1, &gate_tensor);
      set_left_ortho_lim(q1);
      set_right_ortho_lim(q1);

      std::stringstream stream;
      stream << "Error after applying gate \n" << gate << fmt::format("\n to qubits {}.", qubits);
      assert_state_valid(stream.str());
    }

    void reset_from_tensor(const ITensor& tensor) {
      ITensor one_l(left_boundary_index);
      one_l.set(1, 1.0);
      ITensor one_r(right_boundary_index);
      one_r.set(1, 1.0);

      ITensor c = tensor * one_l * one_r;

      for (size_t i = 0; i < num_qubits - 1; i++) {
        std::vector<Index> u_inds = {external_idx(i)};
        if (i > 0) {
          u_inds.push_back(internal_idx(i-1, InternalDir::Right));
        } else {
          u_inds.push_back(left_boundary_index);
        }

        std::vector<Index> v_inds{right_boundary_index};
        for (size_t j = i + 1; j < num_qubits; j++) {
          v_inds.push_back(external_idx(j));
        }
        
        auto [U, S, V] = svd(c, u_inds, v_inds,
            {"Cutoff=",sv_threshold,"MaxDim=",bond_dimension,
            "LeftTags=",fmt::format("Internal,Left,n={}", i),
            "RightTags=",fmt::format("Internal,Right,n={}", i)});

        c = V;
        tensors[i] = U;
        singular_values[i] = S;

        internal_indices[2*i] = commonIndex(U, S);
        internal_indices[2*i + 1] = commonIndex(V, S);
      }

      tensors[num_qubits - 1] = c;
    }

    double trace() const {
      std::vector<ITensor> deltas = get_deltas_between(0, num_blocks());
      ITensor L = left_boundary_tensor(0);
      ITensor R = right_boundary_tensor(num_blocks());
      ITensor contraction = partial_contraction(0, num_blocks(), &deltas, &L, &R);
      return std::abs(tensor_to_scalar(contraction));
    }

    std::vector<ITensor> get_deltas_between(uint32_t i1, uint32_t i2) const {
      std::vector<ITensor> deltas;
      for (size_t i = i1; i < i2; i++) {
        if (qubit_map.contains(i)) {
          uint32_t q = qubit_map.at(i);
          Index idx = external_idx(q);
          deltas.push_back(delta(idx, prime(idx)));
        }
      }

      return deltas;
    }

    double purity() {
      if (is_pure_state()) {
        return 1.0;
      }

      ITensor C = left_environment_tensor(1);
      ITensor L = C;
      L *= conj(prime(prime(C, "Internal"), "Internal"));
      for (size_t i = 1; i < num_blocks(); i++) {
        C = partial_contraction(i, i+1, nullptr, nullptr, nullptr);
        L *= C;
        L *= conj(prime(prime(C, "Internal"), "Internal"));
      }
      L *= delta(right_boundary_index, prime(right_boundary_index), prime(right_boundary_index, 2), prime(right_boundary_index, 3));
      return tensor_to_scalar(L).real();
    }

    size_t bond_dimension_at_site(size_t i) const {
      if (i >= num_qubits - 1) {
        throw std::runtime_error(fmt::format("Cannot check bond dimension of site {} for MPS with {} sites.", i, num_qubits));
      }

      return dim(inds(singular_values[i])[0]);
    }

    void reverse() {
      // Swap constituent tensors
      for (size_t i = 0; i < num_qubits/2; i++) {
        size_t j = num_qubits - i - 1;
        std::swap(tensors[i], tensors[j]);
      }

      for (size_t i = 0; i < blocks.size()/2; i++) {
        size_t j = blocks.size() - i - 1;
        std::swap(blocks[i], blocks[j]);
      }

      for (size_t i = 0; i < singular_values.size()/2; i++) {
        size_t j = num_qubits - i - 1;
        std::swap(singular_values[i], singular_values[j]);
      }

      // Relabel indices
      for (size_t i = 0; i < num_qubits/2; i++) {
        size_t j = num_qubits - i - 1;
        swap_tags(external_indices[i], external_indices[j], tensors[i], tensors[j]);
        std::swap(external_indices[i], external_indices[j]);
      }

      for (size_t i = 0; i < internal_indices.size()/2; i++) {
        size_t j = internal_indices.size() - i - 1;
        std::swap(internal_indices[i], internal_indices[j]);
      }

      // Reform index labels
      for (size_t q = 0; q < qubit_indices.size(); q++) {
        qubit_indices[q] = num_blocks() - qubit_indices[q] - 1;
      }
      std::reverse(qubit_indices.begin(), qubit_indices.end());

      for (size_t b = 0; b < block_indices.size(); b++) {
        block_indices[b] = num_blocks() - block_indices[b] - 1;
      }
      std::reverse(block_indices.begin(), block_indices.end());

      std::map<uint32_t, uint32_t> qubit_map_;
      for (auto const& [i, q] : qubit_map) {
        qubit_map_[num_blocks() - i - 1] = num_qubits - q - 1;
      }
      qubit_map = qubit_map_;

      std::map<uint32_t, uint32_t> block_map_;
      for (auto const& [i, b] : block_map) {
        block_map_[num_blocks() - i - 1] = blocks.size() - b - 1;
      }
      block_map = block_map_;

      // Fix index tags
      for (size_t i = 0; i < internal_indices.size()/2; i++) {
        size_t j = internal_indices.size() - i - 1;
        size_t k1 = i / 2;
        size_t k2 = internal_indices.size() / 2 - k1 - 1;

        Index& i1 = internal_indices[i];
        Index& i2 = internal_indices[j];
        
        swap_tags(i1, i2, singular_values[k1], singular_values[k2]);
        if (k1 == k2) {
          singular_values[k1].swapTags(tags(i1), tags(i2));
        }
      }

      auto swap_internal_indices_at_block = [&](const Index& i1, const Index& i2, size_t k) {
        if (qubit_map.contains(k)) {
          size_t q = qubit_map.at(k);
          tensors[q].swapTags(tags(i1), tags(i2));
        } else {
          size_t b = block_map.at(k);
          blocks[b].swapTags(tags(i1), tags(i2));
          blocks[b].swapPrime(0, 1);
          blocks[b].swapTags(tags(i1), tags(i2));
          blocks[b].swapPrime(0, 1);
        }
      };

      for (size_t i = 0; i < internal_indices.size(); i++) {
        size_t j = internal_indices.size() - i - 1;

        Index i1 = internal_indices[i];
        Index i2 = internal_indices[j];

        size_t k = (i + 1) / 2;
        swap_internal_indices_at_block(i1, i2, k);
      }

      if (num_blocks() % 2) {
        size_t k = num_blocks() / 2;
        size_t i = internal_indices.size()/2;
        size_t j = internal_indices.size() - i - 1;

        swap_internal_indices_at_block(internal_indices[i], internal_indices[j], k);
      }

      std::swap(left_ortho_lim, right_ortho_lim);
      left_ortho_lim = num_qubits - 1 - left_ortho_lim;
      right_ortho_lim = num_qubits - 1 - right_ortho_lim;

      // Fix boundary indices
      std::swap(left_boundary_index, right_boundary_index);
      swap_tags(left_boundary_index, right_boundary_index);
      swap_internal_indices_at_block(left_boundary_index, right_boundary_index, 0);
      swap_internal_indices_at_block(left_boundary_index, right_boundary_index, num_blocks() - 1);
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
        contraction *= singular_values[q - 1];
        contraction *= tensors[q];
        contraction *= prime(other.singular_values[q - 1]);
        contraction *= prime(conj(replaceInds(other.tensors[q], {ext2}, {ext1})), "Internal");
      }

      contraction *= delta(left_boundary_index, prime(other.left_boundary_index));
      contraction *= delta(right_boundary_index, prime(other.right_boundary_index));
      return std::conj(tensor_to_scalar(contraction));
    }

    bool singular_values_trivial(size_t i) const {
      return (dim(internal_idx(i, InternalDir::Right)) == 1)
          && (std::abs(norm(singular_values[i]) - 1.0) < 1e-4);
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
      auto proj = b ? proj1 / std::sqrt(1.0 - prob_zero) : proj0 / std::sqrt(prob_zero);

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

      auto [q1, q2] = to_interval(qubits).value();
      size_t i1 = qubit_indices[q1];
      size_t i2 = qubit_indices[q2 - 1] + 1;

      auto pm = pauli.to_matrix();
      auto id = Eigen::MatrixXcd::Identity(1u << pauli.num_qubits, 1u << pauli.num_qubits);
      Eigen::MatrixXcd proj0 = (id + pm)/2.0;

      double prob_zero = (1.0 + expectation(pauli.superstring(qubits, num_qubits)).real())/2.0;

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

      Eigen::MatrixXcd proj = (m.beta*t).exp();
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
    ITensor orthogonality_tensor_r(uint32_t q) const {
      ITensor A_l = tensors[q];
      size_t i = qubit_indices[q];
      Index left_index = left_boundary_index;
      if (i != 0) {
        left_index = internal_idx(i - 1, InternalDir::Right);
      }

      if (i != num_blocks() - 1) {
        A_l *= singular_values[i];
      }

      return A_l * conj(prime(A_l, left_index));
    }

    ITensor orthogonality_tensor_l(uint32_t q) const {
      ITensor A_r = tensors[q];
      size_t i = qubit_indices[q];
      Index right_index = right_boundary_index;
      if (i != num_blocks() - 1) {
        right_index = internal_idx(i, InternalDir::Left);
      }

      if (i != 0) {
        A_r *= singular_values[i - 1];
      }
      
      return A_r * conj(prime(A_r, right_index));
    }

    std::vector<double> sv_sums() const {
      std::vector<double> sums;
      for (size_t i = 0; i < singular_values.size(); i++) {
        uint32_t N = dim(inds(singular_values[i])[0]);
        double d = 0.0;
        for (uint32_t j = 0; j < N; j++) {
          std::vector<uint32_t> assignments{j + 1, j + 1};
          double c = elt(singular_values[i], assignments);
          d += c*c;
        }
        sums.push_back(d);
      }

      return sums;
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
      //std::cout << fmt::format("Called check_ortho. lims = ({}, {}) orthogonality_level = {}\n", left_ortho_lim, right_ortho_lim, orthogonality_level);
      if (orthogonality_level == 0) {
        return true;
      }

      //std::cout << fmt::format("left lims:\n");
      for (size_t i = 0; i < left_ortho_lim; i++) {
        auto I = orthogonality_tensor_l(i);
        auto d = distance_from_identity(I);
        //std::cout << fmt::format("i = {}, d = {:.5f}\n", i, d);
        if (d > 1e-5) {
          return false;
        }
      }
      //std::cout << fmt::format("right lims:\n");
      for (size_t i = right_ortho_lim + 1; i < num_qubits - 1; i++) {
        auto I = orthogonality_tensor_r(i);
        auto d = distance_from_identity(I);
        //std::cout << fmt::format("i = {}, d = {:.5f}\n", i, d);
        if (d > 1e-5) {
          return false;
        }
      }

      //std::cout << "All good!\n";

      return true;
    }

    void state_checks(const std::string& error_message) const {
      for (size_t i = 0; i < num_blocks() - 1; i++) {
        size_t d = dim(inds(singular_values[i])[0]);
        double s = 0.0;
        for (int j = 1; j <= d; j++) {
          double v = elt(singular_values[i], j, j);
          s += v*v;
          // Valid singular values (between 0 and 1)
          if (v > 1.0 + 1e-8 || v < 0.0 || std::isnan(v)) {
            throw std::runtime_error(fmt::format("{}\nError in singular_values[{}]: {::.5f}\n", error_message, i, singular_values_to_vector(i)));
          }
        }

        // Singular values sum to 1
        if (std::abs(s - 1.0) > 1e-5) {
          throw std::runtime_error(fmt::format("{}\nsingular_values[{}] not normalized: {::.5f}\n", error_message, i, singular_values_to_vector(i)));
        }
      }

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
    // ======================================= DEBUG FUNCTIONS ======================================= //
};

class PauliExpectationTreeImpl {
  public:
    const MatrixProductStateImpl& state;

    Pauli pauli;
    uint8_t phase;

    size_t depth;
    size_t min;
    size_t max;

    bool active;

    // Environment tensors
    ITensor tensor;

    // Pointers for tree traversal
    std::shared_ptr<PauliExpectationTreeImpl> left;
    std::shared_ptr<PauliExpectationTreeImpl> right;

    PauliExpectationTreeImpl(MatrixProductStateImpl& state, const PauliString& p, size_t min, size_t max)
    : state(state), phase(p.get_r()), min(min), max(max) {
      size_t num_qubits = state.num_qubits;
      if (num_qubits != p.num_qubits) {
        throw std::runtime_error(fmt::format("Can't create PauliExpectationTreeImpl; number of qubits does not match: {} and {}.", state.num_qubits, p.num_qubits));
      }

      size_t width = max - min;

      depth = std::ceil(std::log2(width));
      if (is_leaf()) { // depth == 0
        size_t q = min;
        pauli = p.to_pauli(q);
      } else {
        size_t q = width/2;
        left = std::make_shared<PauliExpectationTreeImpl>(state, p, min, min + q);
        right = std::make_shared<PauliExpectationTreeImpl>(state, p, min + q, max);
      }

      update_node();
    }

    bool is_leaf() const {
      return depth == 0;
    }

    bool is_root() const {
      return min == 0 && max == state.num_qubits;
    }

    ITensor partial_contraction(uint32_t q1, uint32_t q2) const {
      if (q1 == min && q2 == max) {
        return tensor;
      }

      if (q2 <= right->min) {
        return left->partial_contraction(q1, q2);
      } else if (q1 >= right->min) {
        return right->partial_contraction(q1, q2);
      } else {
        return left->partial_contraction(q1, left->max) * right->partial_contraction(right->min, q2);
      }
    }

    std::complex<double> partial_expectation(uint32_t q1, uint32_t q2) {
      ITensor contraction = partial_contraction(q1, q2);
      contraction *= state.left_boundary_tensor(left_boundary(q1));
      contraction *= state.right_boundary_tensor(right_boundary(q2));

      return sign_from_bits(phase) * tensor_to_scalar(contraction);
    }

    uint32_t left_boundary(size_t q) const {
      return state.qubit_indices[q];
    }

    uint32_t right_boundary(uint32_t q) const {
      if (q == state.num_qubits) {
        return state.num_blocks();
      } else {
        return state.qubit_indices[q];
      }
    }

    std::pair<uint32_t, uint32_t> get_boundaries() const {
      return {left_boundary(min), right_boundary(max)};
    }

    void update_node() {
      if (is_leaf()) {
        Index idx = state.external_idx(min);
        std::vector<ITensor> p = {pauli_tensor(pauli, prime(idx), idx)};
        auto [i1, i2] = get_boundaries();

        tensor = state.partial_contraction(i1, i2, &p, nullptr, nullptr);
      } else {
        tensor = left->tensor * right->tensor;
      }

      active = false;
    }

    void update_tree() {
      if (active) {
        if (!is_leaf()) {
          left->update_tree();
          right->update_tree();
        }

        update_node();
      }
    }

    uint8_t propogate_pauli(Pauli p, uint32_t q) {
      active = true;

      if (is_leaf()) {
        auto [result, r] = multiply_pauli(p, pauli);
        pauli = result;
        return r;
      } else {
        if (q < right->min) {
          return left->propogate_pauli(p, q);
        } else {
          return right->propogate_pauli(p, q);
        }
      }
    }

    void modify(const PauliString& P) {
      auto support = P.get_support();

      uint8_t phase_change = P.get_r();
      for (const auto q : support) {
        Pauli p = P.to_pauli(q);
        phase_change += propogate_pauli(p, q);
      }

      phase = (phase + phase_change) & 0b11;

      update_tree();
    }

    std::string to_string() const {
      std::string s;
      if (depth == 0) {
        s = std::string(1, pauli_to_char(pauli));
      } else {
        s = left->to_string() + right->to_string();
      }

      if (is_root()) {
        s = PauliString::phase_to_string(phase) + s; 
      }
      
      return s;
    }

    PauliString to_pauli_string() const {
      return PauliString(to_string());
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
  impl->reset_from_tensor(coefficients);
}

MatrixProductState::MatrixProductState()=default;

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

  impl->orthogonalize(0);

  MatrixProductState mps(num_qubits, bond_dimension, sv_threshold);
  mps.impl = std::move(impl);

  return mps;
}

std::string MatrixProductState::to_string() const {
  return impl->to_string();
}

double MatrixProductState::entropy(const Qubits& qubits, uint32_t index) {
  //if (index != 1) {
  //  throw std::runtime_error("Cannot compute Renyi entanglement entropy with index other than 1 for MatrixProductState.");
  //}

	if (qubits.size() == 0) {
		return 0.0;
	}

	std::vector<uint32_t> sorted_qubits(qubits);
	std::sort(sorted_qubits.begin(), sorted_qubits.end());

	if ((sorted_qubits[0] != 0) || !contiguous(sorted_qubits)) {
		throw std::runtime_error("Invalid qubits passed to MatrixProductState.entropy; must be a continuous interval with left side qubit = 0.");
	}

	uint32_t q = sorted_qubits.back() + 1;

	return impl->entropy(q, index);
}

std::vector<double> MatrixProductState::singular_values(uint32_t i) const {
  return impl->singular_values_to_vector(i);
}

std::vector<std::vector<std::vector<std::complex<double>>>> MatrixProductState::tensor(uint32_t q) const {
  return impl->get_tensor(q);
}

double MatrixProductState::magic_mutual_information(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples) {
  if (use_parent) {
    return MagicQuantumState::magic_mutual_information(qubitsA, qubitsB, num_samples);
  }

  if (!contiguous(qubitsA) || !contiguous(qubitsB)) {
    throw std::runtime_error(fmt::format("qubitsA = {}, qubitsB = {} not contiguous. Can't compute MPS.magic_mutual_information.", qubitsA, qubitsB));
  }
  if (!is_bipartition(qubitsA, qubitsB, num_qubits)) {
    throw std::runtime_error(fmt::format("qubitsA = {}, qubitsB = {} are not a bipartition of system with {} qubits. Can't compute MPS.magic_mutual_information.", qubitsA, qubitsB, num_qubits));
  }

  auto pauli_samples = sample_paulis({qubitsA, qubitsB}, num_samples);
  auto data = extract_amplitudes(pauli_samples);
  return MagicQuantumState::calculate_magic_mutual_information_from_samples2(data);
}

double MatrixProductState::magic_mutual_information_montecarlo(
  const Qubits& qubitsA, 
  const Qubits& qubitsB, 
  size_t num_samples, size_t equilibration_timesteps, 
  std::optional<PauliMutationFunc> mutation_opt
) {
  if (use_parent) {
    return MagicQuantumState::magic_mutual_information_montecarlo(qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
  }

  auto prob = [](double t) -> double { return t*t; };

  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);
  std::vector<QubitSupport> supports = {_qubitsA, _qubitsB};
  auto mpo = partial_trace_mps(_qubits);

  auto pauli_samples2 = mpo.sample_paulis_montecarlo(supports, num_samples, equilibration_timesteps, prob, mutation_opt);
  return MagicQuantumState::calculate_magic_mutual_information_from_samples2(extract_amplitudes(pauli_samples2));
}

std::vector<double> MatrixProductState::bipartite_magic_mutual_information(size_t num_samples) { 
  auto pauli_samples = sample_paulis({}, num_samples);
  return impl->process_bipartite_pauli_samples(pauli_samples);
}

std::vector<double> MatrixProductState::bipartite_magic_mutual_information_montecarlo(
  size_t num_samples, size_t equilibration_timesteps, 
  std::optional<PauliMutationFunc> mutation_opt
) {
  if (use_parent) {
    return MagicQuantumState::bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps, mutation_opt);
  }

  auto pauli_samples = sample_paulis_montecarlo({}, num_samples, equilibration_timesteps, [](double t) { return t*t; }, mutation_opt);
  return impl->process_bipartite_pauli_samples(pauli_samples);
}

std::vector<BitAmplitudes> MatrixProductState::sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
  return impl->sample_bitstrings(supports, num_samples);
}

std::vector<PauliAmplitudes> MatrixProductState::sample_paulis(const std::vector<QubitSupport>& supports, size_t num_samples) {
  if (use_parent) {
    return MagicQuantumState::sample_paulis(supports, num_samples);
  }

  return impl->sample_paulis(supports, num_samples);
  //return impl->sample_paulis(qubits, num_samples);
  // Should these checks be done in Impl?
  //std::vector<PauliAmplitudes> samples;
  //if (impl->is_pure_state()) {
  //  samples = impl->sample_paulis(supports, num_samples);
  //} else if (impl->blocks.size() == 1) {
  //  if (impl->block_map.contains(0)) { // Left-bipartite
  //    // TODO check that this is correct!
  //    samples = impl->sample_paulis(supports, num_samples);
  //  } else if (impl->block_map.contains(impl->num_blocks() - 1)) { // Right-bipartite
  //    MatrixProductState reversed(*this);
  //    reversed.reverse();
  //    samples = reversed.impl->sample_paulis(supports, num_samples);
  //  } else {
  //    throw std::runtime_error("Cannot currently perform sample_paulis on non-bipartite mixed states.");
  //  }
  //} else {
  //  throw std::runtime_error("Cannot currently perform sample_paulis on non-bipartite mixed states.");
  //}
}

std::vector<PauliAmplitudes> MatrixProductState::sample_paulis_montecarlo(const std::vector<QubitSupport>& supports, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt) {
  if (use_parent) {
    return MagicQuantumState::sample_paulis_montecarlo(supports, num_samples, equilibration_timesteps, prob, mutation_opt);
  }

  int i = impl->orthogonality_level;
  impl->set_orthogonality_level(0);

  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  PauliString p(num_qubits);
  PauliExpectationTree tree(*this, p);

  auto samples = impl->sample_paulis_montecarlo(tree, supports, num_samples, equilibration_timesteps, prob, mutation);
  impl->set_orthogonality_level(i);
  return samples;
}

std::complex<double> MatrixProductState::expectation(const PauliString& p) const {
  return impl->expectation(p);
}

std::complex<double> MatrixProductState::expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const {
  return impl->expectation(m, qubits);
}

double MatrixProductState::expectation(const BitString& bits, std::optional<QubitSupport>& support) const {
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

void MatrixProductState::reverse() {
  impl->reverse();
}

void MatrixProductState::orthogonalize(uint32_t q) {
  impl->orthogonalize(q);
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

// ----------------------------------------------------------------------- //
// --------------- PauliExpectationTree implementation ------------------- //
// ----------------------------------------------------------------------- //

PauliExpectationTree::PauliExpectationTree(const MatrixProductState& state, const PauliString& p) : num_qubits(state.num_qubits) {
  impl = std::make_unique<PauliExpectationTreeImpl>(*state.impl, p, 0, num_qubits);
}

PauliExpectationTree::~PauliExpectationTree()=default;

std::complex<double> PauliExpectationTree::expectation() const {
  return impl->partial_expectation(0, num_qubits);
}

std::complex<double> PauliExpectationTree::partial_expectation(uint32_t q1, uint32_t q2) const {
  return impl->partial_expectation(q1, q2);
}

void PauliExpectationTree::propogate_pauli(Pauli p, uint32_t q) {
  impl->propogate_pauli(p, q);
}

void PauliExpectationTree::modify(const PauliString& p) {
  impl->modify(p);
}

std::string PauliExpectationTree::to_string() const {
  return impl->to_string();
}

PauliString PauliExpectationTree::to_pauli_string() const {
  return impl->to_pauli_string();
}

bool inspect_svd_error() {
  ITensor theta;
  readFromFile("theta.h5", theta);


  std::vector<Index> u_inds{findIndex(theta, "i=94"), findIndex(theta, "n=93")};
  std::vector<Index> v_inds{findIndex(theta, "i=95"), findIndex(theta, "n=95")};

  print(theta);

  size_t q = 94;

  size_t bond_dimension = 128;
  double threshold = 1e-4;
  auto [U, S, V] = svd(theta, u_inds, v_inds, 
      {"Cutoff=",threshold,"MaxDim=",bond_dimension,
      "LeftTags=",fmt::format("n={}", q),
      "RightTags=",fmt::format("n={}", q+1)});
  PrintData(S);
  return true;
}

int load_seed(const std::string& filename) {
  auto e = MatrixProductStateImpl::read_svd_error(filename);
  return e.seed;
}
