#include "LinearCode.h"

ParityCheckMatrix::ParityCheckMatrix(uint32_t num_rows, uint32_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

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

  if (num_cols < num_rows) {
    throw std::invalid_argument("Invalid parity check matrix.");
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



GeneratorMatrix::GeneratorMatrix(uint32_t num_rows, uint32_t num_cols) : BinaryMatrix(num_rows, num_cols) { }

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

std::vector<uint32_t> GeneratorMatrix::generator_locality_samples(const std::vector<size_t>& sites) {
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

  return {partial_rank(sites), partial_rank(sites_complement), rank(), num_rows, num_cols};
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
