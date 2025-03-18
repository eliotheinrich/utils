#pragma once

#include "BinaryMatrix.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <optional>

class BinaryPolynomialTerm {
  public:
    std::vector<size_t> inds;
    bool coefficient;

    BinaryPolynomialTerm() : coefficient(true) {}

    BinaryPolynomialTerm(const std::vector<size_t>& inds, bool coefficient=true) : inds(inds), coefficient(coefficient) {}

    BinaryPolynomialTerm(const BinaryPolynomialTerm& term) : inds(term.inds), coefficient(term.coefficient) {}

    std::string to_string() const {
      if (degree() == 0) {
        return coefficient ? "1" : "0";
      }

      std::string s = "";
      for (size_t i = 0; i < degree(); i++) {
        s += "x_" + std::to_string(inds[i]);
        if (i != inds.size() - 1) {
          s += " ";
        }
      }

      return s;
    }

    bool evaluate(const BitString& z) const {
      bool b = coefficient;

      for (auto const i : inds) {
        b &= z.get(i);
      }

      return b;
    }

    BinaryPolynomialTerm reduce() const {
      std::vector<size_t> new_inds;
    
      for (auto const i : inds) {
        if (std::find(new_inds.begin(), new_inds.end(), i) == new_inds.end()) {
          new_inds.push_back(i);
        }
      }

      return BinaryPolynomialTerm(new_inds);
    }

    BinaryPolynomialTerm partial_evaluate(const BitString& bits, const std::vector<size_t>& applied_inds) const {
      std::vector<size_t> new_inds;
      bool b = true;

      for (const auto i : inds) {
        auto found = std::find(applied_inds.begin(), applied_inds.end(), i);
        // bit i is not being assigned to
        if (found == applied_inds.end()) {
          new_inds.push_back(i);
        } else {
          size_t j = std::distance(applied_inds.begin(), found);
          b &= bits.get(j);
        }
      }

      return BinaryPolynomialTerm(new_inds, b);
    } 

    uint32_t degree() const {
      return inds.size();
    }

    bool operator==(const BinaryPolynomialTerm& rhs) const {
      if (degree() != rhs.degree()) {
        return false;
      }

      if (coefficient != rhs.coefficient) {
        return false;
      }

      for (const auto i : inds) {
        if (std::find(rhs.inds.begin(), rhs.inds.end(), i) == rhs.inds.end()) {
          return false;
        }
      }

      for (const auto i : rhs.inds) {
        if (std::find(inds.begin(), inds.end(), i) == inds.end()) {
          return false;
        }
      }

      return true;
    }

    bool contains_ind(size_t i) const {
      return std::find(inds.begin(), inds.end(), i) != inds.end();
    }
};

class BinaryPolynomial {
  public:
    size_t n;
    std::vector<BinaryPolynomialTerm> terms;

    BinaryPolynomial() {}

    BinaryPolynomial(size_t n) : n(n) {}

    BinaryPolynomial(const std::vector<BinaryPolynomialTerm>& terms, size_t n) : n(n) {
      for (auto const& term : terms) {
        for (auto const& i : term.inds) {
          if (i < 0 || i >= n) {
            throw std::invalid_argument("Invalid terms passed to BinaryPolynomial constructor.");
          }
        }

        add_term(term);
      }
    }

    std::optional<size_t> contains_term(const BinaryPolynomialTerm& term) const {
      for (size_t i = 0; i < terms.size(); i++) {
        if (term == terms[i]) {
          return i;
        }
      }

      return std::nullopt;
    }

    bool add_term(const BinaryPolynomialTerm& term) {
      if (!term.coefficient) {
        return false;
      }

      auto term_ind = contains_term(term);
      if (term_ind.has_value()) {
        size_t i = term_ind.value();
        terms.erase(terms.begin() + i);
        return false;
      } else {
        terms.push_back(term);
        return true;
      }
    }

    bool add_term(const std::vector<size_t>& inds) {
      return add_term(BinaryPolynomialTerm(inds));
    }

    // Convenience functions up to deg-3 terms
    bool add_term(size_t i) {
      std::vector<size_t> inds{i};
      return add_term(inds);
    }

    bool add_term(size_t i, size_t j) {
      std::vector<size_t> inds{i, j};
      return add_term(inds);
    }
    
    bool add_term(size_t i, size_t j, size_t k) {
      std::vector<size_t> inds{i, j, k};
      return add_term(inds);
    }

    std::string to_string() const {
      std:: string s = "";
      for (size_t i = 0; i < terms.size(); i++) {
        s += terms[i].to_string();
        if (i != terms.size() - 1) {
          s += " + ";
        }
      }

      return s;
    }

    bool evaluate(BitString z) const {
      bool s = false;
      for (auto const& term : terms) {
        s ^= term.evaluate(z);
      }

      return s;
    }

    BinaryPolynomial partial_evaluate(const BitString& bits, const std::vector<size_t>& assigned_bits) const {
      BinaryPolynomial poly(n);

      for (auto const& term : terms) {
        BinaryPolynomialTerm new_term = term.partial_evaluate(bits, assigned_bits);
        poly.add_term(new_term);
      }

      return poly;
    }

    double partition_function() const {
      double z = 0.0;
      uint64_t s = 1u << n;

      for (uint64_t i = 0; i < s; i++) {
        z += std::pow(-1.0, evaluate(BitString::from_bits(n, i)));
      }

      return z/s;
    }


    BinaryPolynomial operator+(const BinaryPolynomial& other) {
      BinaryPolynomial poly(n);
      for (auto const& term : terms) {
        poly.add_term(term);
      }

      for (auto const& term : other.terms) {
        poly.add_term(term);
      }

      return poly;
    }
};


