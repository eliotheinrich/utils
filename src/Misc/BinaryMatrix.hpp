#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#define binary_word uint32_t

struct Bitstring {
  std::vector<binary_word> bits;
  uint32_t num_bits;

  Bitstring(size_t num_bits) : num_bits(num_bits) {
    bits = std::vector<binary_word>(num_bits / 32u + 1);
  }

  Bitstring(const std::vector<binary_word>& bits, uint32_t num_bits) : bits(bits), num_bits(num_bits) {}

  Bitstring(const std::vector<bool>& bits_bool) : num_bits(bits_bool.size()) {
    size_t num_words = bits_bool.size() / 32u + 1;
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

  std::string to_string() const {
    std::string s;
    for (size_t i = 0; i < num_bits; i++) {
      s += std::to_string(get(i));
    }

    return s;
  }

  inline void set(size_t i, bool v) {
    size_t word_ind = i / 32u;
    size_t bit_ind = i % 32u;
    bits[word_ind] = (bits[word_ind] & ~(1u << bit_ind)) | (v << bit_ind);
  }

  inline bool get(size_t i) const {
    size_t word_ind = i / 32u;
    size_t bit_ind = i % 32u;
    return static_cast<bool>((bits[word_ind] >> bit_ind) & 1u);
  }
};

class BinaryMatrix {
  public:
    BinaryMatrix(size_t num_rows, size_t num_cols) : num_rows(num_rows), num_cols(num_cols), num_words(num_cols / 32u + 1) {
      data = std::vector<Bitstring>(num_rows, Bitstring(num_cols));
    }

    BinaryMatrix() : BinaryMatrix(0, 0) {}

    BinaryMatrix(const std::vector<Bitstring>& data) : data(data) {
      num_rows = data.size();
      if (num_rows == 0) {
        num_words = 0;
        num_cols = 0;
      } else {
        size_t num_bits = data[0].num_bits;
        num_cols = num_bits / 32u + 1;
        for (size_t i = 1; i < num_rows; i++) {
          if (data[i].num_bits != num_words) {
            throw std::invalid_argument("Provided data is ragged!");
          }
        }
      }
    }

    BinaryMatrix(const std::vector<Bitstring>& data, size_t num_cols) : BinaryMatrix(data) {
      if ((num_words - 1)*32u > num_cols || num_cols > num_words*32u) {
        throw std::invalid_argument("Provided number of columns is not valid for the data.");
      }

      this->num_cols = num_cols;
    }

    std::string to_string() const {
      std::string s;
      for (size_t i = 0; i < num_rows; i++) {
        if (i != 0) {
          s += " "; 
        }
        s += "[ ";
        for (size_t j = 0; j < num_cols; j++) {
          s += std::to_string(static_cast<uint8_t>(get(i, j))) + " ";
        }
        s += "]";
        if (i != num_rows - 1) {
          s += ",\n";
        }
      }

      return "[" + s + "]";
    }

    void rref() {
      size_t c = 0;
      size_t i = 0;
      while (c < num_cols) {
        size_t r = i;

        bool found_pivot = false;
        while (r < num_rows) {
          if (get(r, c)) {
            found_pivot = true;
            break;
          }
          r++;
        }

        if (found_pivot) {
          swap_rows(r, i);
          for (size_t ri = 0; ri < num_rows; ri++) {
            if (ri != i && get(ri, c)) {
              add_rows(i, ri);
            }
          }
          i++;
        }
        c++;
      }
    }

    uint32_t rank(bool inplace=false) {
      BinaryMatrix copy;
      BinaryMatrix& M = inplace ? *this : copy;

      if (!inplace) {
        copy = BinaryMatrix(data);
        M = copy;
      }

      M.rref();
      uint32_t r = 0;
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_words; j++) {
          if (M.data[i].get(j)) {
            r++;
            break;
          }
        }
      }
      
      return r;
    }

    std::vector<bool> solve_linear_system(const std::vector<bool>& v) {
      BinaryMatrix copy(*this);
      copy.append_col(v);
      copy.rref();

      std::vector<bool> solution(num_rows);
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
          if (copy.get(i, j)) {
            solution[i] = copy.get(i, num_cols);
          }
        }
      }

      return solution;
    }

    bool in_col_space(const std::vector<bool>& v) const {
      BinaryMatrix copy(data);

      uint32_t r1 = copy.rank();
      copy.append_col(v);
      uint32_t r2 = copy.rank(true);

      return r1 == r2;
    }

    BinaryMatrix transpose() const {
      BinaryMatrix copy(num_cols, num_rows);
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
          copy.set(j, i, get(i, j));
        }
      }
      
      return copy;
    }

    inline bool get(size_t i, size_t j) const {
      return data[i].get(j);
    }

    inline void set(size_t i, size_t j, bool v) {
      data[i].set(j, v);
    }

    void swap_rows(size_t r1, size_t r2) {
      std::swap(data[r1], data[r2]);
    }

    void add_rows(size_t r1, size_t r2) {
      for (size_t i = 0; i < num_words; i++) {
        data[r2].set(i, data[r2].get(i) ^ data[r1].get(i));
      }
    }

    void append_row(const std::vector<bool>& row) {
      size_t row_num_words = row.size() / 32u + 1;
      if (row_num_words != num_words) {
        throw std::invalid_argument("Invalid row length.");
      }

      std::vector<binary_word> row_words(row_num_words);

      for (size_t i = 0; i < row.size(); i++) {
        size_t word_ind = i / 32u;
        size_t bit_ind = i % 32u;
        row_words[word_ind] = (row_words[word_ind] & ~(1u << bit_ind)) | (row[i] << bit_ind);
      }
      
      num_rows++;
      data.push_back(Bitstring(row_words, num_cols));
    }

    void append_col(const std::vector<bool>& col) {
      if (col.size() != data.size()) {
        throw std::invalid_argument("Invalid column length.");
      }
      
      // Could be optimized...
      *this = transpose();
      append_row(col);
      *this = transpose();
    }
    
    bool operator[](const size_t i, const size_t j) const {
      return get(i, j);
    }


  private:
    size_t num_rows;
    size_t num_cols;
    size_t num_words;
    std::vector<Bitstring> data;
};
