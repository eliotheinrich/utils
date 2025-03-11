#include "LinearCode.h"


ParityCheckMatrix::ParityCheckMatrix(size_t num_rows, size_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

ParityCheckMatrix::ParityCheckMatrix(const BinaryMatrix& other) : ParityCheckMatrix(other.num_rows, other.num_cols) {
  data = other.data;
}

GeneratorMatrix ParityCheckMatrix::to_generator_matrix(bool inplace) {
  ParityCheckMatrix workspace;
  if (inplace) {
    workspace = *this;
  } else {
    workspace = ParityCheckMatrix(*this);
  }

  // Put parity check matrix in canonical form
  workspace.reduce();
  size_t n = workspace.num_cols;
  size_t m = workspace.num_rows;
  size_t k = n - m;

  std::vector<size_t> sites(m);
  std::iota(sites.begin(), sites.end(), k);
  workspace.partial_rref(sites);

  std::unique_ptr<BinaryMatrixBase> A_ptr = workspace.slice(0, m, 0, k);
  BinaryMatrix* A = dynamic_cast<BinaryMatrix*>(A_ptr.get());

  GeneratorMatrix G(n, k);
  for (size_t i = 0; i < k; i++) {
    G.set(i, i, 1);
  }

  for (size_t i = k; i < n; i++) {
    G.data[i] = A->data[i - k];
  }

  G.transpose();
  return G;
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

std::pair<std::optional<size_t>, std::vector<size_t>> ParityCheckMatrix::leaf_removal_iteration(std::minstd_rand& rng) {
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

  size_t r = rng() % leafs.size();
  remove_row(edges[r]);

  return {edges[r], sizes};
}

bool ParityCheckMatrix::congruent(const GeneratorMatrix& G) const {
  GeneratorMatrix copy(G);
  copy.transpose();

  auto K = multiply(copy);
  for (size_t r = 0; r < K.num_rows; r++) {
    for (size_t c = 0; c < K.num_cols; c++) {
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

bool ParityCheckMatrix::is_in_space(const std::vector<bool>& v) const {
  std::vector<bool> parity = BinaryMatrixBase::multiply(v);
  
  int pc = 0;
  for (auto b : parity) {
    pc = (pc + int(b)) % 2;
  }

  return !bool(pc);
}



GeneratorMatrix::GeneratorMatrix(size_t num_rows, size_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

GeneratorMatrix::GeneratorMatrix(const BinaryMatrix& other) : GeneratorMatrix(other.num_rows, other.num_cols) {
  data = other.data;
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
  for (size_t i = 0; i < num_cols - num_rows; i++) {
    At->append_row(I.data[i]);
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

  auto K = H.multiply(copy);
  for (size_t r = 0; r < K.num_rows; r++) {
    for (size_t c = 0; c < K.num_cols; c++) {
      if (K.get(r, c)) {
        return false;
      }
    }
  }

  return true;
}
