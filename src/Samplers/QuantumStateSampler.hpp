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

      sample_configurational_entropy = dataframe::utils::get<int>(params, "sample_configurational_entropy", false);
      configurational_entropy_method = dataframe::utils::get<std::string>(params, "configurational_entropy_method", "sampled");
      if (configurational_entropy_method != "sampled" && configurational_entropy_method != "exhaustive") {
        throw std::runtime_error(fmt::format("Invalid configurational entropy method: {}\n", configurational_entropy_method));
      }
      num_configurational_entropy_samples = dataframe::utils::get<int>(params, "num_configurational_entropy_samples", 1000);

      sample_configurational_mutual_entropy = dataframe::utils::get<int>(params, "sample_configurational_mutual_entropy", false);

      sample_spin_glass_order = dataframe::utils::get<int>(params, "sample_spin_glass_order", false);
      if (sample_spin_glass_order) {
        spin_glass_order_assume_symmetry = dataframe::utils::get<int>(params, "spin_glass_order_assume_symmetry", false);
        spin_glass_order_basis = parse_basis(dataframe::utils::get<std::string>(params, "spin_glass_order_basis", "Z"));
      }
    }

    ~QuantumStateSampler()=default;

    void add_probability_samples(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) {
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

    void add_bitstring_distribution(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) {
      if (state->get_num_qubits() > 31) {
        throw std::runtime_error("Cannot generate bitstring distribution for n > 31 qubits.");
      }

      std::vector<double> probabilities = state->probabilities();
      dataframe::utils::emplace(samples, "bitstring_distribution", probabilities);
    }

    void add_configurational_entropy_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) {
      std::vector<double> Ws;
      if (configurational_entropy_method == "sampled") {
        auto bitstrings = state->sample_bitstring_amplitudes(num_configurational_entropy_samples);
        Ws = std::vector<double>(bitstrings.size());
        size_t n = 0;
        for (const auto prob : bitstrings) {
          Ws[n++] = -std::log(prob);
        }
      } else {
        auto probs = state->probabilities();
        double w = 0.0;
        for (auto p : probs) {
          if (p > 1e-10) {
            w -= p * std::log(p);
          }
        }

        Ws = {w};
      }

      std::vector<std::vector<double>> Wss = {Ws};
      dataframe::utils::emplace(samples, "configurational_entropy", Wss);
    }

    void add_configurational_mutual_entropy_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) {
      size_t nsamples = num_configurational_entropy_samples;
      auto estimate_configurational_entropy = [nsamples](const std::shared_ptr<QuantumState>& state) {
        std::vector<double> samples = state->sample_bitstring_amplitudes(nsamples);
        std::vector<double> log_samples(samples.size());
        
        for (size_t i = 0; i < samples.size(); i++) {
          log_samples[i] = -std::log(samples[i]);
        }

        return dataframe::Sample(log_samples).get_mean();
      };

      size_t nqb = state->get_num_qubits();

      Qubits qubitsA(nqb/2);
      std::iota(qubitsA.begin(), qubitsA.end(), 0);

      Qubits qubitsB(nqb/2);
      std::iota(qubitsB.begin(), qubitsB.end(), nqb/2);

      auto stateA = state->partial_trace(qubitsB);
      auto stateB = state->partial_trace(qubitsA);

      double WA = estimate_configurational_entropy(stateA);
      double WB = estimate_configurational_entropy(stateB);
      double W = estimate_configurational_entropy(state);

      dataframe::utils::emplace(samples, "mutual_configurational_entropy", WA + WB - W);
    }

    void add_spin_glass_order_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) {
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

      if (sample_configurational_entropy) {
        add_configurational_entropy_samples(samples, state);
      }

      if (sample_configurational_mutual_entropy) {
        add_configurational_mutual_entropy_samples(samples, state);
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

    bool sample_configurational_entropy;
    std::string configurational_entropy_method;
    size_t num_configurational_entropy_samples;

    bool sample_configurational_mutual_entropy;

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

