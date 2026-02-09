#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <memory>
#include <random>

#include <qutils/QuantumCircuit.h>
#include <fmt/format.h>

class BinaryMatrixBase {
  public:
    BinaryMatrixBase(uint32_t num_rows, uint32_t num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    virtual ~BinaryMatrixBase()=default;

    uint32_t get_num_rows() const {
      return num_rows;
    }

    uint32_t get_num_cols() const {
      return num_cols;
    }

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


    virtual std::vector<std::pair<size_t, size_t>> rref() {
      std::vector<size_t> sites(num_cols);
      std::iota(sites.begin(), sites.end(), 0);
      return partial_rref(sites);
    }

    virtual std::vector<std::pair<size_t, size_t>> partial_rref(const std::vector<size_t>& sites) {
      std::vector<std::pair<size_t, size_t>> pivots;

      size_t c = 0;
      size_t i = 0;
      while (c < sites.size()) {
        size_t ci = sites[c];
        size_t r = i;

        bool found_pivot = false;
        while (r < num_rows) {
          if (get(r, ci)) {
            found_pivot = true;
            break;
          }
          r++;
        }

        if (found_pivot) {
          if (r != i) {
            swap_rows(r, i);
            pivots.push_back({r, i});
          }
          for (size_t ri = 0; ri < num_rows; ri++) {
            if (ri != i && get(ri, ci)) {
              add_rows(i, ri);
            }
          }
          i++;
        }
        c++;
      }
      
      return pivots;
    }

    virtual std::unique_ptr<BinaryMatrixBase> slice(size_t r1, size_t r2, size_t c1, size_t c2) const=0;

    virtual BitString row(size_t r) const {
      BitString row(num_cols);
      for (size_t i = 0; i < num_cols; i++) {
        row[i] = get(r, i);
      }
      
      return row;
    }

    virtual BitString col(size_t c) const {
      BitString col(num_rows);
      for (size_t i = 0; i < num_rows; i++) {
        col[i] = get(i, c);
      }

      return col;
    }

    virtual uint32_t partial_rank(const std::vector<size_t>& sites, bool inplace=false) {
      BinaryMatrixBase* workspace;
      std::unique_ptr<BinaryMatrixBase> dummy;
      if (inplace) {
        workspace = this;
      } else {
        dummy = clone();
        workspace = dummy.get();
      }

      workspace->partial_rref(sites);
      uint32_t r = 0;
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < sites.size(); j++) {
          if (workspace->get(i, sites[j])) {
            r++;
            break;
          }
        }
      }
      
      return r;
    }

    virtual uint32_t rank(bool inplace=false) {
      std::vector<size_t> sites(num_cols);
      std::iota(sites.begin(), sites.end(), 0);
      return partial_rank(sites, inplace);
    }

    virtual BitString multiply(const BitString& v) const {
      BitString result(num_rows);

      if (v.get_num_bits() != num_cols) {
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

    BitString solve_linear_system(const BitString& v) {
      std::unique_ptr<BinaryMatrixBase> copy = clone();
      copy->append_col(v);
      copy->rref();

      BitString solution(num_rows);
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

    bool in_col_space(const BitString& v) const {
      std::unique_ptr<BinaryMatrixBase> copy = clone();
      copy->transpose();
      return copy->in_row_space(v);
    }

    bool in_row_space(const BitString& v) const {
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

    virtual void swap_cols(size_t c1, size_t c2) {
      for (size_t r = 0; r < num_rows; ++r) {
        bool v1 = get(r, c1);
        set(r, c1, get(r, c2));
        set(r, c2, v1);
      }
    }

    virtual void add_rows(size_t r1, size_t r2)=0;

    virtual void append_row(const BitString& row)=0;

    virtual void set_row(size_t r, const BitString& row)=0;

    virtual void append_col(const BitString& col) {
      if (col.get_num_bits() != num_rows) {
        throw std::invalid_argument("Invalid column length.");
      }
      
      // Could be optimized...
      transpose();
      append_row(col);
      transpose();
    }

    virtual void remove_row(size_t r)=0;

    virtual void remove_col(size_t c) {
      if (c >= num_cols) {
        throw std::invalid_argument(fmt::format("Cannot delete {}; outside of range.", c));
      }

      transpose();
      remove_row(c);
      transpose();
    }

    virtual std::unique_ptr<BinaryMatrixBase> clone() const=0;

  protected:
    uint32_t num_rows;
    uint32_t num_cols;
};
