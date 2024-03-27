#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <memory>
#include <random>
#include <optional>

class BinaryMatrixBase {
  public:
    uint32_t num_rows;
    uint32_t num_cols;
    BinaryMatrixBase(uint32_t num_rows, uint32_t num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    virtual std::string to_string() const {
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


    virtual void rref() {
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

    virtual uint32_t rank(std::optional<BinaryMatrixBase*> A = std::nullopt) {
      BinaryMatrixBase* workspace;
      if (A.has_value()) {
        workspace = A.value();
      } else {
        workspace = this;
      }

      workspace->rref();
      uint32_t r = 0;
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
          if (workspace->get(i, j)) {
            r++;
            break;
          }
        }
      }
      
      return r;
    }

    std::vector<bool> multiply(const std::vector<bool>& v) const {
      std::vector<bool> result(num_rows);

      if (v.size() != num_cols) {
        throw std::invalid_argument("Dimension mismatch in BinaryMatrix multiply.");
      }

      for (size_t i = 0; i < num_rows; i++) {
        result[i] = false;
        for (size_t j = 0; j < num_cols; j++) {
          result[i] = result[i] ^ (get(i, j) && v[j]);
        }
      }

      return result;
    }

    std::vector<bool> solve_linear_system(const std::vector<bool>& v) {
      std::unique_ptr<BinaryMatrixBase> copy = clone();
      copy->append_col(v);
      copy->rref();

      std::vector<bool> solution(num_rows);
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
          if (copy->get(i, j)) {
            solution[j] = copy->get(i, num_cols);
            break;
          }
        }
      }

      return solution;
    }

    bool in_col_space(const std::vector<bool>& v) const {
      std::unique_ptr<BinaryMatrixBase> copy = clone();
      copy->transpose();
      return copy->in_row_space(v);
    }

    bool in_row_space(const std::vector<bool>& v) const {
      std::unique_ptr<BinaryMatrixBase> copy = clone();
      uint32_t r1 = copy->rank();
      copy->append_row(v);
      uint32_t r2 = copy->rank();

      return r1 == r2;
    }

    virtual void transpose()=0;

    virtual bool get(size_t i, size_t j) const=0;

    virtual void set(size_t i, size_t j, bool v)=0;

    virtual void swap_rows(size_t r1, size_t r2)=0;

    virtual void add_rows(size_t r1, size_t r2)=0;

    virtual void append_row(const std::vector<bool>& row)=0;

    virtual void append_col(const std::vector<bool>& col) {
      if (col.size() != num_rows) {
        throw std::invalid_argument("Invalid column length.");
      }
      
      // Could be optimized...
      transpose();
      append_row(col);
      transpose();
    }

    virtual std::unique_ptr<BinaryMatrixBase> clone() const=0;
};
