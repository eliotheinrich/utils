#pragma once

#include "BinaryMatrix.hpp"
#include <algorithm>
#include <set>
#include <iterator>
#include <vector>

class SparseBinaryMatrix : public BinaryMatrixBase {
  private:
    std::vector<std::set<size_t>> inds;

  public:
    SparseBinaryMatrix(size_t num_rows, size_t num_cols) : BinaryMatrixBase(num_rows, num_cols) {
      inds = std::vector<std::set<size_t>>(num_rows);
    }

    SparseBinaryMatrix() : SparseBinaryMatrix(0, 0) {}
    
    SparseBinaryMatrix(const std::vector<std::set<size_t>>& inds, size_t num_cols) : BinaryMatrixBase(inds.size(), num_cols), inds(inds) {}

    virtual std::string to_string() const override {
      std::string s = "";
      bool first = true;
      for (size_t i = 0; i < num_rows; i++) {
        for (const auto& c : inds[i]) {
          if (!first) {
            s += ", ";
          }
          s += "(" + std::to_string(i) + ", " + std::to_string(c) + ")";
          first = false;
        }
      }

      return s;
    }

    virtual bool get(size_t i, size_t j) const override {
      return inds[i].contains(j);
    }

    virtual void set(size_t i, size_t j, bool v) override {
      if (get(i, j) == v) {
        return;
      }

      if (get(i, j)) {
        inds[i].erase(j);
      } else {
        inds[i].insert(j);
      }
    }

    virtual void swap_rows(size_t r1, size_t r2) override {
      std::swap(inds[r1], inds[r2]);
    }

    virtual void add_rows(size_t r1, size_t r2) override {
      std::set<size_t> diff;
      std::set_symmetric_difference(
        inds[r1].begin(), inds[r1].end(),
        inds[r2].begin(), inds[r2].end(),
        std::inserter(diff, diff.begin())
      );

      inds[r2] = diff;
    }

    virtual void append_row(const std::vector<bool>& row) override {
      if (row.size() != num_cols) {
        throw std::invalid_argument("Invalid row length.");
      }

      std::set<size_t> new_row;

      for (size_t i = 0; i < row.size(); i++) {
        if (row[i]) {
          new_row.insert(i);
        }
      }

      inds.push_back(new_row);
      num_rows++;
    }

    virtual std::unique_ptr<BinaryMatrixBase> clone() const override {
      return std::make_unique<SparseBinaryMatrix>(inds, num_cols);
    }

    virtual void transpose() override {
      std::vector<std::set<size_t>> new_inds(num_cols);

      for (size_t i = 0; i < num_rows; i++) {
        for (const auto& c : inds[i]) {
          new_inds[c].insert(i);
        }
      }

      inds = new_inds;
      std::swap(num_rows, num_cols);
    }
};
