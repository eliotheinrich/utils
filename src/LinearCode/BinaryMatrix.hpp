#pragma once

#include "BinaryMatrixBase.hpp"
#include <Random.hpp>

#include <fmt/format.h>

class BinaryMatrix : public BinaryMatrixBase {
  public:
    BinaryMatrix(size_t num_rows, size_t num_cols) : BinaryMatrixBase(num_rows, num_cols) {
      data = std::vector<BitString>(num_rows, BitString(num_cols));
    }

    BinaryMatrix() : BinaryMatrix(0, 0) {}

    BinaryMatrix(const std::vector<BitString>& data) : BinaryMatrixBase(data.size(), BinaryMatrix::extract_num_cols(data)), data(data) {
      if (num_rows != 0) {
        for (size_t i = 1; i < num_rows; i++) {
          if (data[i].get_num_bits() != num_cols) {
            throw std::invalid_argument("Provided data is ragged!");
          }
        }
      }
    }

    BinaryMatrix(const std::vector<BitString>& data, size_t num_cols) : BinaryMatrix(data) {
      this->num_cols = num_cols;

      if ((num_words() - 1)*binary_word_size() > num_cols || num_cols > num_words()*binary_word_size()) {
        std::cout << "num_words() = " << num_words() << ", binary_word_size() = " << binary_word_size() << ", num_cols = " << num_cols << ", data.size() = " << data.size() << "\n";
        throw std::invalid_argument("Provided number of columns is not valid for the data.");
      }
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
      std::vector<BitString> data_new(num_cols, BitString(num_rows));
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

    virtual BitString row(size_t r) const override {
      return data[r];
    }

    virtual void append_row(const BitString& row) override {
      if (row.get_num_bits() != num_cols) {
        throw std::invalid_argument("Invalid row length.");
      }

      num_rows++;
      data.push_back(row);
    }

    virtual void set_row(size_t r, const BitString& row) override {
      if (row.get_num_bits() != num_cols) {
        throw std::invalid_argument("Invalid row length.");
      }

      data[r] = row;
    }

    virtual void remove_row(size_t r) override {
      if (r >= num_rows) {
        throw std::invalid_argument(fmt::format("Cannot delete {}; outside of range.", r));
      }

      data.erase(data.begin() + r);
      num_rows--;
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

    BinaryMatrix mmultiply(const BinaryMatrix& other) const {
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

    struct BitRef {

      BitRef(BitString& row, size_t i) : row_(row), i_(i) {}
      BitRef& operator=(bool v) { 
        row_.set(i_, v); 
        return *this; 

      }
      operator bool() const { 
        return row_.get(i_); 
      }
      private:
        BitString& row_;
        size_t i_;
    };

    struct RowRef {
      RowRef(BitString* row) : row_(row) {}

      // Bit proxy
      struct BitRef {
        BitRef(BitString* row, size_t i) : row_(row), i_(i) {}
        BitRef& operator=(bool v) {
          row_->set(i_, v);
          return *this;
        }
        operator bool() const {
          return row_->get(i_);
        }
        private:
        BitString* row_;
        size_t i_;
      };

      // Access individual bits
      BitRef operator[](size_t i) { return BitRef(row_, i); }

      // Assign another row proxy (bit-by-bit copy)
      RowRef& operator=(const RowRef& other) {
        if (row_ == other.row_) return *this;  // self-assign
        for (size_t i = 0; i < other.row_->size(); ++i)
          (*this)[i] = (*other.row_)[i];
        return *this;
      }

      private:
      BitString* row_;
    };

    struct ConstRowRef {
      ConstRowRef(const BitString* row) : row_(row) {}

      struct BitRef {
        BitRef(const BitString* row, size_t i) : row_(row), i_(i) {}

        // Only allow reading, no assignment
        operator bool() const {
          return row_->get(i_);
        }

        private:
        const BitString* row_;
        size_t i_;
      };

      BitRef operator[](size_t i) const { return BitRef(row_, i); }

      private:
        const BitString* row_;
    };

    RowRef operator[](size_t r) { 
      return RowRef(&data[r]); 
    }

    const ConstRowRef operator[](size_t r) const { 
      return ConstRowRef(&data[r]); 
    }

    const std::vector<BitString>& get_data() const { 
      return data; 
    }

  protected:
    std::vector<BitString> data;

    size_t num_words() const {
      return num_cols / binary_word_size() + 1;
    }

    static size_t extract_num_cols(const std::vector<BitString>& data) {
      size_t num_rows = data.size();
      if (num_rows == 0) {
        return 0;
      } else {
        return data[0].get_num_bits();
      }
    }
};
