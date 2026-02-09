#include "LinearCode.h"


ParityCheckMatrix::ParityCheckMatrix(size_t num_rows, size_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

ParityCheckMatrix::ParityCheckMatrix(const BinaryMatrix& other) : ParityCheckMatrix(other.get_num_rows(), other.get_num_cols()) {
  data = other.get_data();
}

GeneratorMatrix ParityCheckMatrix::to_generator_matrix(bool inplace) {
  // 1. Workspace copy
  ParityCheckMatrix Hsys = inplace ? *this : ParityCheckMatrix(*this);

  const size_t m = Hsys.get_num_rows();
  const size_t n = Hsys.get_num_cols();
  const size_t k = n - m;

  // 2. Bring H to systematic form [A | I_m]
  std::vector<size_t> col_perm(n);
  std::iota(col_perm.begin(), col_perm.end(), 0);

  for (size_t r = 0; r < m; ++r) {
    // Find pivot in row r
    size_t pivot_col = r;  // start searching from current row index
    while (pivot_col < n && Hsys.get(r, pivot_col) == 0) ++pivot_col;
    if (pivot_col == n)
      throw std::runtime_error("ParityCheckMatrix is rank-deficient");

    size_t target_col = k + r;
    if (pivot_col != target_col) {
      // Swap columns in Hsys
      Hsys.swap_cols(pivot_col, target_col);
      std::swap(col_perm[pivot_col], col_perm[target_col]);
    }

    // Eliminate other 1s in this column
    for (size_t rr = 0; rr < m; ++rr) {
      if (rr != r && Hsys.get(rr, target_col)) {
        Hsys.add_rows(rr, r);
      }
    }
  }

  // 3. Extract A block (m x k)
  auto A_ptr = Hsys.slice(0, m, 0, k);
  BinaryMatrixBase* A = A_ptr.get();  // work via base class
  if (!A) throw std::runtime_error("Failed to extract A block");

  // 4. Build G = [I_k | A^T]
  GeneratorMatrix G(k, n);
  for (size_t i = 0; i < k; ++i) {
    G.set(i, i, 1);  // identity
    for (size_t j = 0; j < m; ++j)
      G.set(i, k + j, A->get(j, i));  // transpose A
  }

  // 5. Undo column permutation
  GeneratorMatrix G_out(k, n);
  for (size_t j = 0; j < n; ++j) {
    size_t orig_col = col_perm[j];
    for (size_t i = 0; i < k; ++i) {
      G_out.set(i, orig_col, G.get(i, j));
    }
  }

  G_out.rref();
  return G_out;
}

size_t ParityCheckMatrix::degree(size_t c) const {
  size_t n = 0;
  for (size_t i = 0; i < num_rows; i++) {
    if (get(i, c)) {
      n++;
    }
  }

  return n;
}

std::vector<size_t> ParityCheckMatrix::degree_distribution() const {
  std::vector<size_t> sizes(num_rows, 0u);
  for (size_t i = 0; i < num_cols; i++) {
    sizes[degree(i)]++;
  }

  return sizes;
}

std::pair<std::optional<size_t>, std::vector<size_t>> ParityCheckMatrix::leaf_removal_iteration() {
  std::vector<size_t> sizes(num_rows+1);
  std::vector<size_t> leafs;
  std::vector<size_t> edges;

  for (size_t c = 0; c < num_cols; c++) {
    size_t deg = 0;
    size_t e;
    for (size_t r = 0; r < num_rows; r++) {
      if (get(r, c)) {
        e = r;
        deg++;
      }
    }

    if (deg == 1) {
      leafs.push_back(c);
      edges.push_back(e);
    }

    sizes[deg]++;
  }

  if (leafs.size() == 0) {
    return {std::nullopt, sizes};
  }

  size_t r = randi() % leafs.size();
  remove_row(edges[r]);

  return {edges[r], sizes};
}

bool ParityCheckMatrix::congruent(const GeneratorMatrix& G) const {
  GeneratorMatrix copy(G);
  copy.transpose();

  auto K = mmultiply(copy);
  for (size_t r = 0; r < K.get_num_rows(); r++) {
    for (size_t c = 0; c < K.get_num_cols(); c++) {
      if (K.get(r, c)) {
        return false;
      }
    }
  }

  return true;
}

void ParityCheckMatrix::reduce() {
  rref();

  std::vector<size_t> to_remove;

  for (size_t i = 0; i < num_rows; i++) {
    bool remove = true;
    for (size_t j = 0; j < num_cols; j++) {
      if (get(i, j)) {
        remove = false;
        break;
      }
    }

    if (remove) {
      to_remove.push_back(i);
    }
  }

  std::reverse(to_remove.begin(), to_remove.end());
  for (size_t r : to_remove) {
    remove_row(r);
  }
}

bool ParityCheckMatrix::is_in_space(const BitString& v) const {
  BitString parity = multiply(v);
  
  int pc = 0;
  for (size_t i = 0; i < parity.get_num_bits(); ++i) {
    pc = (pc + int(parity[i])) % 2;
  }

  return !bool(pc);
}



GeneratorMatrix::GeneratorMatrix(size_t num_rows, size_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

GeneratorMatrix::GeneratorMatrix(const BinaryMatrix& other) : GeneratorMatrix(other.get_num_rows(), other.get_num_cols()) {
  data = other.get_data();
}

ParityCheckMatrix GeneratorMatrix::to_parity_check_matrix(bool inplace) {
  BinaryMatrix workspace;
  if (inplace) {
    workspace = *this;
  } else {
    workspace = BinaryMatrix(*this);
  }

  std::vector<size_t> sites(num_rows);
  std::iota(sites.begin(), sites.end(), 0);
  workspace.partial_rref(sites);

  std::unique_ptr<BinaryMatrixBase> A = workspace.slice(0, num_rows, num_rows, num_cols);
  BinaryMatrix* At = dynamic_cast<BinaryMatrix*>(A.release());

  BinaryMatrix I = BinaryMatrix::identity(num_cols - num_rows);
  const std::vector<BitString>& I_data = I.get_data();
  for (size_t i = 0; i < num_cols - num_rows; i++) {
    At->append_row(I_data[i]);
  }

  At->transpose();
  return ParityCheckMatrix(*At);
}

uint32_t GeneratorMatrix::sym(const std::vector<size_t>& sites1, const std::vector<size_t>& sites2) {
  std::vector<size_t> all_sites;
  all_sites.insert(all_sites.end(), sites1.begin(), sites1.end());
  all_sites.insert(all_sites.end(), sites2.begin(), sites2.end());

  GeneratorMatrix G1 = supported(sites1);
  GeneratorMatrix G2 = supported(sites2);
  GeneratorMatrix G3 = supported(all_sites);

  return G1.rank() + G2.rank() - G3.rank();
}

GeneratorMatrix GeneratorMatrix::truncate(const std::vector<size_t>& rows) const {
  GeneratorMatrix G(rows.size(), num_cols);
  for (size_t i = 0; i < rows.size(); i++) {
    G.data[i] = data[rows[i]];
  }

  return G;
}

GeneratorMatrix GeneratorMatrix::supported(const std::vector<size_t>& sites) const {
  std::vector<size_t> has_support;
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < sites.size(); j++) {
      if (get(i, sites[j])) {
        has_support.push_back(i);
        break;
      }
    }
  }

  return truncate(has_support);
}


uint32_t GeneratorMatrix::generator_locality(const std::vector<size_t>& sites) {
  std::vector<size_t> sites_complement;
  std::vector<bool> exists(num_cols, false);
  for (size_t i = 0; i < sites.size(); i++) {
    if (sites[i] >= num_cols) {
      throw std::invalid_argument("Invalid site index.");
    }

    exists[sites[i]] = true;
  }

  for (size_t i = 0; i < num_cols; i++) {
    if (!exists[i]) {
      sites_complement.push_back(i);
    }
  }

  return partial_rank(sites) + partial_rank(sites_complement) - rank();
}

bool GeneratorMatrix::congruent(const ParityCheckMatrix& H) const {
  GeneratorMatrix copy(*this);
  copy.transpose();

  auto K = H.mmultiply(copy);
  for (size_t r = 0; r < K.get_num_rows(); r++) {
    for (size_t c = 0; c < K.get_num_cols(); c++) {
      if (K.get(r, c)) {
        return false;
      }
    }
  }

  return true;
}
