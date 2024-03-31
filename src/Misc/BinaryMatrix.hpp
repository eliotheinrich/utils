#pragma once

#include "BinaryMatrixBase.hpp"

#define binary_word uint32_t

constexpr uint32_t binary_word_size() {
  return 8*sizeof(static_cast<binary_word>(0));
}


struct Bitstring {
  std::vector<binary_word> bits;
  uint32_t num_bits;

  Bitstring(size_t num_bits) : num_bits(num_bits) {
    bits = std::vector<binary_word>(num_bits / binary_word_size() + 1);
  }

  Bitstring(const std::vector<binary_word>& bits, uint32_t num_bits) : bits(bits), num_bits(num_bits) {}

  Bitstring(const std::vector<bool>& bits_bool) : num_bits(bits_bool.size()) {
    size_t num_words = bits_bool.size() / binary_word_size() + 1;
    bits = std::vector<binary_word>(num_words);
    for (size_t i = 0; i < bits_bool.size(); i++) {
      set(i, bits_bool[i]);
    }
  }

  Bitstring(uint64_t z, uint32_t num_bits) : num_bits(num_bits) {
    if (num_bits < 32) {
      bits = std::vector<binary_word>(1, 0u);
    } else if (num_bits < 64) {
      bits = std::vector<binary_word>(2, 0u);
    } else {
      throw std::invalid_argument("Too many bits for a 64bit integer.");
    }

    for (size_t i = 0; i < num_bits; i++) {
      set(i, (z >> i) & 1u);
    }
  }

  Bitstring(const Bitstring& bitstring) : Bitstring(bitstring.bits, bitstring.num_bits) {}

  static Bitstring random(uint32_t num_bits, std::minstd_rand& rng) {
    size_t num_words = num_bits / binary_word_size() + 1;
    std::vector<binary_word> bits(num_words);
    for (size_t i = 0; i < bits.size(); i++) {
      bits[i] = rng();
    }

    return Bitstring(bits, num_bits);
  }

  std::string to_string() const {
    std::string s;
    for (size_t i = 0; i < num_bits; i++) {
      s += std::to_string(static_cast<uint8_t>(get(i)));
    }

    return s;
  }

  inline void set(size_t i, bool v) {
    if (i > num_bits) {
      throw std::invalid_argument("Invalid bit index.");
    }

    size_t word_ind = i / binary_word_size();
    size_t bit_ind = i % binary_word_size();
    bits[word_ind] = (bits[word_ind] & ~(1u << bit_ind)) | (v << bit_ind);
  }

  inline bool get(size_t i) const {
    if (i > num_bits) {
      throw std::invalid_argument("Invalid bit index.");
    }

    size_t word_ind = i / binary_word_size();
    size_t bit_ind = i % binary_word_size();
    return static_cast<bool>((bits[word_ind] >> bit_ind) & 1u);
  }

  uint32_t hamming_weight() const {
    uint32_t s = 0;
    for (size_t i = 0; i < num_bits; i++) {
      s += get(i);
    }

    return s;
  }

  Bitstring operator^(const Bitstring& other) const {
    if (num_bits != other.num_bits) {
      throw std::invalid_argument("Bitstring size mismatch.");
    }

    Bitstring new_bits(num_bits);
    for (size_t i = 0; i < bits.size(); i++) {
      new_bits.bits[i] = bits[i] ^ other.bits[i];
    }

    return new_bits;
  }
};

class BinaryMatrix : public BinaryMatrixBase {
  public:
    BinaryMatrix(size_t num_rows, size_t num_cols) : BinaryMatrixBase(num_rows, num_cols) {
      data = std::vector<Bitstring>(num_rows, Bitstring(num_cols));
    }

    BinaryMatrix() : BinaryMatrix(0, 0) {}

    BinaryMatrix(const std::vector<Bitstring>& data) : BinaryMatrixBase(data.size(), BinaryMatrix::extract_num_cols(data)), data(data) {
      if (num_rows != 0) {
        for (size_t i = 1; i < num_rows; i++) {
          if (data[i].num_bits != num_cols) {
            throw std::invalid_argument("Provided data is ragged!");
          }
        }
      }
    }

    BinaryMatrix(const std::vector<Bitstring>& data, size_t num_cols) : BinaryMatrix(data) {
      if ((num_words() - 1)*binary_word_size() > num_cols || num_cols > num_words()*binary_word_size()) {
        throw std::invalid_argument("Provided number of columns is not valid for the data.");
      }

      this->num_cols = num_cols;
    }

    BinaryMatrix(const BinaryMatrix& other) : BinaryMatrix(other.data, other.num_cols) {}

    static BinaryMatrix identity(size_t n) {
      BinaryMatrix I(n, n);
      for (size_t i = 0; i < n; i++) {
        I.set(i, i, 1);
      }

      return I;
    }

    virtual void transpose() override {
      std::vector<Bitstring> data_new(num_cols, Bitstring(num_rows));
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
          data_new[j].set(i, get(i, j));
        }
      }

      data = data_new;
      std::swap(num_rows, num_cols);
    }

    virtual bool get(size_t i, size_t j) const override {
      return data[i].get(j);
    }

    virtual void set(size_t i, size_t j, bool v) override {
      data[i].set(j, v);
    }

    virtual void swap_rows(size_t r1, size_t r2) override {
      std::swap(data[r1], data[r2]);
    }

    virtual void add_rows(size_t r1, size_t r2) override {
      data[r2] = data[r1] ^ data[r2];
    }

    virtual void append_row(const std::vector<bool>& row) override {
      size_t row_num_words = row.size() / binary_word_size() + 1;
      if (row_num_words != num_words()) {
        throw std::invalid_argument("Invalid row length.");
      }

      std::vector<binary_word> row_words(row_num_words);

      for (size_t i = 0; i < row.size(); i++) {
        size_t word_ind = i / binary_word_size();
        size_t bit_ind = i % binary_word_size();
        row_words[word_ind] = (row_words[word_ind] & ~(1u << bit_ind)) | (row[i] << bit_ind);
      }

      num_rows++;
      data.push_back(Bitstring(row_words, num_cols));
    }

    void append_row(const Bitstring& row) {
      size_t row_num_bits = row.num_bits;
      if (row_num_bits != num_cols) {
        throw std::invalid_argument("Invalid row length.");
      }

      num_rows++;
      data.push_back(row);
    }

    virtual std::unique_ptr<BinaryMatrixBase> slice(size_t r1, size_t r2, size_t c1, size_t c2) const override {
      if (r2 < r1 || c2 < c1) {
        throw std::invalid_argument("Invalid slice indices.");
      }

      std::unique_ptr<BinaryMatrix> A = std::make_unique<BinaryMatrix>(r2 - r1, c2 - c1);
      for (size_t r = 0; r < r2 - r1; r++) {
        for (size_t c = 0; c < c2 - c1; c++) {
          A->set(r, c, get(r + r1, c + c1));
        }
      }

      return A;
    }

    virtual std::unique_ptr<BinaryMatrixBase> clone() const override {
      return std::make_unique<BinaryMatrix>(data);
    }

    virtual std::unique_ptr<BinaryMatrixBase> to_generator_matrix(bool inplace=false) override {
      BinaryMatrix workspace;
      if (inplace) {
        workspace = *this;
      } else {
        workspace = BinaryMatrix(*this);
      }

      if (num_cols <= num_rows) {
        throw std::invalid_argument("Invalid parity check matrix.");
      }

      size_t k = num_cols - num_rows;

      std::vector<size_t> sites(num_rows);
      std::iota(sites.begin(), sites.end(), k);
      workspace.partial_rref(sites);

      std::unique_ptr<BinaryMatrixBase> A_ptr = workspace.slice(0, num_rows, 0, k);
      BinaryMatrix* A = dynamic_cast<BinaryMatrix*>(A_ptr.get());

      std::unique_ptr<BinaryMatrix> G = std::make_unique<BinaryMatrix>(num_cols, k);
      for (size_t i = 0; i < k; i++) {
        G->set(i, i, 1);
      }

      for (size_t i = k; i < num_cols; i++) {
        G->data[i] = A->data[i - k];
      }

      G->transpose();
      return G;
    }

    virtual std::unique_ptr<BinaryMatrixBase> to_parity_check_matrix(bool inplace=false) override {
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
      BinaryMatrix* At = dynamic_cast<BinaryMatrix*>(A.get());

      BinaryMatrix I = BinaryMatrix::identity(num_cols - num_rows);
      for (size_t i = 0; i < num_cols - num_rows; i++) {
        At->append_row(I.data[i]);
      }

      A->transpose();
      return A;
    }

    BinaryMatrix multiply(const BinaryMatrix& other) const {
      if (num_cols != other.num_rows) {
        throw std::invalid_argument("BinaryMatrix dimension mismatch.");
      }

      BinaryMatrix A(num_rows, other.num_cols);

      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < other.num_cols; j++) {
          bool v = false;
          for (size_t k = 0; k < num_cols; k++) {
            v ^= get(i, k) && other.get(k, j);
          }
          A.set(i, j, v);
        }
      }

      return A;
    }

  private:
    size_t num_words() const {
      return num_cols / binary_word_size() + 1;
    }

    static size_t extract_num_cols(const std::vector<Bitstring>& data) {
      size_t num_rows = data.size();
      if (num_rows == 0) {
        return 0;
      } else {
        return data[0].num_bits;
      }
    }

    std::vector<Bitstring> data;
};
