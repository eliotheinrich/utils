#pragma once

#include <Frame.h>
#include "QuantumState.h"
#include "QuantumCircuit.h"

#include <iostream>
#include <string>

enum Basis {
  x, y, z
};

static inline Basis parse_basis(const std::string& s) {
  if (s.size() != 1) {
    throw std::runtime_error(fmt::format("Invalid string {} provided as basis.", s));
  }

  if (s[0] == 'X' || s[0] == 'x') {
    return Basis::x;
  } else if (s[0] == 'Y' || s[0] == 'y') {
    return Basis::y;
  } else if (s[0] == 'Z' || s[0] == 'z') {
    return Basis::z;
  } else {
    throw std::runtime_error(fmt::format("Invalid string {} provided as basis.", s));
  }
}

class QuantumStateSampler {
  public:
    QuantumStateSampler(dataframe::ExperimentParams& params) {
      num_bins = dataframe::utils::get<int>(params, "num_bins", 100);
      min_prob = dataframe::utils::get<double>(params, "min_prob", 0.0);
      max_prob = dataframe::utils::get<double>(params, "max_prob", 1.0);
      sample_probabilities = dataframe::utils::get<int>(params, "sample_probabilities", false);

      if (max_prob <= min_prob) {
        throw std::invalid_argument("max_prob must be greater than min_prob.");
      }

      sample_bitstring_distribution = dataframe::utils::get<int>(params, "sample_bitstring_distribution", false);

      sample_spin_glass_order = dataframe::utils::get<int>(params, "sample_spin_glass_order", false);
      if (sample_spin_glass_order) {
        spin_glass_order_assume_symmetry = dataframe::utils::get<int>(params, "spin_glass_order_assume_symmetry", false);
        spin_glass_order_basis = parse_basis(dataframe::utils::get<std::string>(params, "spin_glass_order_basis", "Z"));
      }
    }

    ~QuantumStateSampler()=default;

    void add_probability_samples(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) const {
      if (state->get_num_qubits() > 31) {
        throw std::runtime_error("Cannot generate probabilities for n > 31 qubits.");
      }

      std::vector<double> probabilities = state->probabilities();
      size_t s = probabilities.size();

      std::vector<uint32_t> probability_counts(num_bins);
      uint32_t num_counts = 0;
      for (uint32_t i = 0; i < s; i++) {
        double p = probabilities[i];
        if (p >= min_prob && p < max_prob) {
          size_t idx = get_bin_idx(probabilities[i]);
          probability_counts[idx]++;
          num_counts++;
        }
      }


      std::vector<double> probability_probs(num_bins, 0.0);
      
      if (num_counts != 0) {
        for (uint32_t i = 0; i < num_bins; i++) {
          probability_probs[i] = static_cast<double>(probability_counts[i])/num_counts;
        }
      }

      dataframe::utils::emplace(samples, "probabilities", probability_probs);
    }

    void add_bitstring_distribution(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) const {
      if (state->get_num_qubits() > 31) {
        throw std::runtime_error("Cannot generate bitstring distribution for n > 31 qubits.");
      }

      std::vector<double> probabilities = state->probabilities();
      dataframe::utils::emplace(samples, "bitstring_distribution", probabilities);
    }

    void add_spin_glass_order_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      size_t num_qubits = state->get_num_qubits();
      double O = 0.0;

      Basis basis = spin_glass_order_basis;

      // Returns a PauliString with P at each specified site, where P = X, Y, Z depending on
      // the spin_glass_order_basis.
      auto make_pauli = [num_qubits, basis](const std::vector<size_t>& sites) {
        PauliString P(num_qubits);
        switch (basis) {
          case Basis::x:
            for (const auto q : sites) {
              P.set_x(q, 1);
            }
            break;
          case Basis::y:
            for (const auto q : sites) {
              P.set_x(q, 1);
              P.set_z(q, 1);
            }
            break;
          case Basis::z:
            for (const auto q : sites) {
              P.set_z(q, 1);
            }
            break;
        }

        return P;
      };

      std::vector<double> c(num_qubits);
      for (size_t i = 0; i < num_qubits; i++) {
        if (spin_glass_order_assume_symmetry) {
          c[i] = 0.0;
        } else {
          PauliString Pi = make_pauli({i});
          c[i] = std::pow(std::abs(state->expectation(Pi)), 2.0);
        }
      }

      for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
          PauliString Pij = make_pauli({i, j});

          auto cij = std::pow(std::abs(state->expectation(Pij)), 2.0);

          O += cij - c[i]*c[j];
        }
      }

      dataframe::utils::emplace(samples, "spin_glass_order", O/num_qubits);
    }

    void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) {
      if (sample_probabilities) {
        add_probability_samples(samples, state);
      }

      if (sample_bitstring_distribution) {
        add_bitstring_distribution(samples, state);
      }
      
      if (sample_spin_glass_order) {
        add_spin_glass_order_samples(samples, state);
      }
    }

  private:	
    uint32_t num_bins;
    double min_prob;
    double max_prob;
    bool sample_probabilities;

    bool sample_bitstring_distribution;

    bool sample_spin_glass_order;
    bool spin_glass_order_assume_symmetry;
    Basis spin_glass_order_basis;

    uint32_t get_bin_idx(double s) const {
      if ((s < min_prob) || (s > max_prob)) {
        std::string error_message = std::to_string(s) + " is not between " + std::to_string(min_prob) + " and " + std::to_string(max_prob) + ". \n";
        throw std::invalid_argument(error_message);
      }

      double bin_width = static_cast<double>(max_prob - min_prob)/num_bins;
      uint32_t idx = static_cast<uint32_t>((s - min_prob) / bin_width);
      return idx;
    }
};

