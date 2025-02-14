#pragma once

#include <Frame.h>
#include "QuantumState.h"
#include "QuantumCircuit.h"

#include <iostream>
#include <string>

class QuantumStateSampler {
  public:
    enum sre_method_t {
      Virtual, MonteCarlo, Exhaustive, Exact
    };

    sre_method_t parse_sre_method(const std::string& s) {
      if (s == "montecarlo") {
        return sre_method_t::MonteCarlo;
      } else if (s == "virtual") {
        return sre_method_t::Virtual;
      } else if (s == "exhaustive") {
        return sre_method_t::Exhaustive;
      } else if (s == "exact") {
        return sre_method_t::Exact;
      } else {
        throw std::runtime_error(fmt::format("Stabilizer renyi entropy method \"{}\" not found.", s));
      }
    }

    QuantumStateSampler(dataframe::ExperimentParams& params) {
      system_size = dataframe::utils::get<int>(params, "system_size");

      num_bins = dataframe::utils::get<int>(params, "num_bins", 100);
      min_prob = dataframe::utils::get<double>(params, "min_prob", 0.0);
      max_prob = dataframe::utils::get<double>(params, "max_prob", 1.0);
      sample_probabilities = dataframe::utils::get<int>(params, "sample_probabilities", false);

      if (max_prob <= min_prob) {
        throw std::invalid_argument("max_prob must be greater than min_prob.");
      }

      sample_bitstring_distribution = dataframe::utils::get<int>(params, "sample_bitstring_distribution", false);

      sre_use_parent_methods = dataframe::utils::get<int>(params, "sre_use_parent_methods", false);

      sample_stabilizer_renyi_entropy = dataframe::utils::get<int>(params, "sample_stabilizer_renyi_entropy", false);
      save_sre_samples = dataframe::utils::get<int>(params, "save_sre_samples", 0);
      sre_num_samples = dataframe::utils::get<int>(params, "sre_num_samples", 1000);
      sre_method = parse_sre_method(dataframe::utils::get<std::string>(params, "sre_method", "virtual"));
      sre_mc_equilibration_timesteps = dataframe::utils::get<int>(params, "sre_mc_equilibration_timesteps", 5*system_size);

      if (sample_stabilizer_renyi_entropy) {
        renyi_indices = parse_renyi_indices(dataframe::utils::get<std::string>(params, "stabilizer_renyi_indices", "1"));
      }

      sample_magic_mutual_information = dataframe::utils::get<int>(params, "sample_magic_mutual_information", false);
      if (sample_magic_mutual_information) {
        magic_mutual_information_subsystem_size = dataframe::utils::get<int>(params, "magic_mutual_information_subsystem_size", system_size/4);
        if (magic_mutual_information_subsystem_size > system_size/2) {
          throw std::runtime_error(fmt::format("magic_mutual_information_subsystem_size = {}, but system_size = {}", magic_mutual_information_subsystem_size, system_size));
        }
        subsystem_offset_A = dataframe::utils::get<int>(params, "subsystem_offset_A", 0);
        subsystem_offset_B = dataframe::utils::get<int>(params, "subsystem_offset_B", system_size - magic_mutual_information_subsystem_size);
      }

      sample_bipartite_magic_mutual_information = dataframe::utils::get<int>(params, "sample_bipartite_magic_mutual_information", false);
      if (sample_magic_mutual_information || sample_bipartite_magic_mutual_information) {
        save_mmi_samples = dataframe::utils::get<int>(params, "save_mmi_samples", false);
      }
    }

    ~QuantumStateSampler()=default;

    void set_montecarlo_update(PauliMutationFunc f) {
      mutation = f;
    }

    void add_probability_samples(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) {
      if (state->num_qubits > 31) {
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
      if (state->num_qubits > 31) {
        throw std::runtime_error("Cannot generate bitstring distribution for n > 31 qubits.");
      }

      std::vector<double> probabilities = state->probabilities();
      dataframe::utils::emplace(samples, "bitstring_distribution", probabilities);
    }

    void add_mmi_samples(dataframe::SampleMap &samples, const std::vector<MutualMagicData>& data) const {
      std::vector<std::vector<double>> tA2;
      std::vector<std::vector<double>> tB2;
      std::vector<std::vector<double>> tAB2;

      std::vector<std::vector<double>> tA4;
      std::vector<std::vector<double>> tB4;
      std::vector<std::vector<double>> tAB4;

      for (auto const [t2, t4] : data) {
        auto [tA2_, tB2_, tAB2_] = unfold_mutual_magic_amplitudes(t2);
        auto [tA4_, tB4_, tAB4_] = unfold_mutual_magic_amplitudes(t4);

        tA2.push_back(tA2_);
        tB2.push_back(tB2_);
        tAB2.push_back(tAB2_);
        tA4.push_back(tA4_);
        tB4.push_back(tB4_);
        tAB4.push_back(tAB4_);
      }

      dataframe::utils::emplace(samples, "magic_mutual_information_tA2", tA2);
      dataframe::utils::emplace(samples, "magic_mutual_information_tB2", tB2);
      dataframe::utils::emplace(samples, "magic_mutual_information_tAB2", tAB2);
      dataframe::utils::emplace(samples, "magic_mutual_information_tA4", tA4);
      dataframe::utils::emplace(samples, "magic_mutual_information_tB4", tB4);
      dataframe::utils::emplace(samples, "magic_mutual_information_tAB4", tAB4);
    }

    void add_samples(dataframe::SampleMap &samples, const std::shared_ptr<QuantumState>& state) {
      if (sample_probabilities) {
        add_probability_samples(samples, state);
      }

      if (sample_bitstring_distribution) {
        add_bitstring_distribution(samples, state);
      }

      auto compute_sre_montecarlo = [&](std::shared_ptr<QuantumState> state, const std::vector<size_t>& indices, const std::vector<PauliAmplitudes>& pauli_samples) {
        std::vector<double> amplitudes = extract_amplitudes(pauli_samples)[0];
        std::vector<double> stabilizer_renyi_entropy;
        for (auto index : indices) {
          double M = state->stabilizer_renyi_entropy(index, amplitudes);
          stabilizer_renyi_entropy.push_back(M);
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

      auto compute_sre_exhaustive = [&](std::shared_ptr<QuantumState> state, const std::vector<size_t>& indices, const std::vector<PauliAmplitudes>& pauli_samples) {
        std::vector<double> amplitudes = extract_amplitudes(pauli_samples)[0];
        std::vector<double> stabilizer_renyi_entropy;
        double purity = std::pow(2.0, state->num_qubits) * state->purity();
        for (auto index : indices) {
          double M = 0.0;
          if (index == 1) {
            for (auto p : amplitudes) {
              double prob = p*p/purity;
              if (p > 1e-6) {
                M += prob * std::log(prob);
              }
            }
            M = -M;
          } else {
            for (auto p : amplitudes) {
              double prob = p*p/purity;
              M += std::pow(prob, index);
            }
            M = std::log(M)/(1.0 - index);
          }

          stabilizer_renyi_entropy.push_back(M);
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

      state->set_use_parent_implementation(sre_use_parent_methods);

      if (sample_stabilizer_renyi_entropy) {
        ProbabilityFunc prob = [](double t) -> double { return t*t; };
        std::vector<double> stabilizer_renyi_entropy;
        std::vector<double> amplitudes;
        if (sre_method == sre_method_t::Exhaustive) {
          auto pauli_samples = state->sample_paulis_exhaustive({});
          std::tie(amplitudes, stabilizer_renyi_entropy) = compute_sre_exhaustive(state, renyi_indices, pauli_samples);
        } else if (sre_method == sre_method_t::MonteCarlo) {
          auto pauli_samples = state->sample_paulis_montecarlo({}, sre_num_samples, sre_mc_equilibration_timesteps, prob, mutation);
          std::tie(amplitudes, stabilizer_renyi_entropy) = compute_sre_montecarlo(state, renyi_indices, pauli_samples);
        } else if (sre_method == sre_method_t::Exact) {
          auto pauli_samples = state->sample_paulis_exact({}, sre_num_samples, prob);
          std::tie(amplitudes, stabilizer_renyi_entropy) = compute_sre_montecarlo(state, renyi_indices, pauli_samples);
        } else if (sre_method == sre_method_t::Virtual) {
          auto pauli_samples = state->sample_paulis({}, sre_num_samples);
          std::tie(amplitudes, stabilizer_renyi_entropy) = compute_sre_montecarlo(state, renyi_indices, pauli_samples);
        }

        if (save_sre_samples) {
          dataframe::utils::emplace(samples, "stabilizer_renyi_entropy_amplitudes", amplitudes);
        }

        for (size_t i = 0; i < renyi_indices.size(); i++) {
          dataframe::utils::emplace(samples, fmt::format("stabilizer_renyi_entropy{}", renyi_indices[i]), stabilizer_renyi_entropy[i]);
        }
      }

      if (sample_magic_mutual_information) {
        std::vector<uint32_t> qubitsA(magic_mutual_information_subsystem_size);
        std::vector<uint32_t> qubitsB(magic_mutual_information_subsystem_size);
        for (size_t i = 0; i < magic_mutual_information_subsystem_size; i++) {
          qubitsA[i] = i + subsystem_offset_A;
          qubitsB[i] = i + subsystem_offset_B;
        }

        double mmi_sample;
        if (sre_method == sre_method_t::Exhaustive) {
          mmi_sample = state->magic_mutual_information_exhaustive(qubitsA, qubitsB);
        } else if (sre_method == sre_method_t::MonteCarlo) {
          if (save_mmi_samples) {
            auto mmi_data = state->magic_mutual_information_samples_montecarlo(qubitsA, qubitsB, sre_num_samples, sre_mc_equilibration_timesteps, mutation);
            add_mmi_samples(samples, {mmi_data});
            mmi_sample = QuantumState::calculate_magic_mutual_information_from_samples(mmi_data);
          } else {
            mmi_sample = state->magic_mutual_information_montecarlo(qubitsA, qubitsB, sre_num_samples, sre_mc_equilibration_timesteps, mutation);
          }
        } else if (sre_method == sre_method_t::Exact) {
          if (save_mmi_samples) {
            auto mmi_data = state->magic_mutual_information_samples_exact(qubitsA, qubitsB, sre_num_samples);
            add_mmi_samples(samples, {mmi_data});
            mmi_sample = QuantumState::calculate_magic_mutual_information_from_samples(mmi_data);
          } else {
            mmi_sample = state->magic_mutual_information_exact(qubitsA, qubitsB, sre_num_samples);
          }
        } else if (sre_method == sre_method_t::Virtual) {
          mmi_sample = state->magic_mutual_information(qubitsA, qubitsB, sre_num_samples);
        }

        dataframe::utils::emplace(samples, "magic_mutual_information", mmi_sample);
      }

      if (sample_bipartite_magic_mutual_information) {
        std::vector<double> mmi_samples;

        if (sre_method == sre_method_t::Exhaustive) {
          mmi_samples = state->bipartite_magic_mutual_information_exhaustive();
        } else if (sre_method == sre_method_t::MonteCarlo) {
          if (save_mmi_samples) {
            auto mmi_data = state->bipartite_magic_mutual_information_samples_montecarlo(sre_num_samples, sre_mc_equilibration_timesteps, mutation);
            add_mmi_samples(samples, mmi_data);
            mmi_samples.resize(mmi_data.size());
            std::transform(mmi_data.begin(), mmi_data.end(), mmi_samples.begin(), [](const MutualMagicData& s) { return QuantumState::calculate_magic_mutual_information_from_samples(s); });
          } else {
            mmi_samples = state->bipartite_magic_mutual_information_montecarlo(sre_num_samples, sre_mc_equilibration_timesteps, mutation);
          }
        } else if (sre_method == sre_method_t::Exact) {
          if (save_mmi_samples) {
            auto mmi_data = state->bipartite_magic_mutual_information_samples_exact(sre_num_samples);
            add_mmi_samples(samples, mmi_data);
            mmi_samples.resize(mmi_data.size());
            std::transform(mmi_data.begin(), mmi_data.end(), mmi_samples.begin(), [](const MutualMagicData& s) { return QuantumState::calculate_magic_mutual_information_from_samples(s); });
          } else {
            mmi_samples = state->bipartite_magic_mutual_information_exact(sre_num_samples);
          }
        } else if (sre_method == sre_method_t::Virtual) {
          mmi_samples = state->bipartite_magic_mutual_information(sre_num_samples);
        }

        dataframe::utils::emplace(samples, "bipartite_magic_mutual_information", mmi_samples);
      }
    }

  private:	
    uint32_t system_size;

    uint32_t num_bins;
    double min_prob;
    double max_prob;
    bool sample_probabilities;

    bool sample_bitstring_distribution;

    bool sre_use_parent_methods;

    bool sample_stabilizer_renyi_entropy;
    std::vector<size_t> renyi_indices;

    sre_method_t sre_method;
    bool save_sre_samples;
    size_t sre_num_samples;
    size_t sre_mc_equilibration_timesteps;

    bool sample_magic_mutual_information;
    bool save_mmi_samples;
    size_t magic_mutual_information_subsystem_size;
    size_t subsystem_offset_A;
    size_t subsystem_offset_B;
    bool sample_bipartite_magic_mutual_information;

    std::optional<PauliMutationFunc> mutation;

    uint32_t get_bin_idx(double s) const {
      if ((s < min_prob) || (s > max_prob)) {
        std::string error_message = std::to_string(s) + " is not between " + std::to_string(min_prob) + " and " + std::to_string(max_prob) + ". \n";
        throw std::invalid_argument(error_message);
      }

      double bin_width = static_cast<double>(max_prob - min_prob)/num_bins;
      uint32_t idx = static_cast<uint32_t>((s - min_prob) / bin_width);
      return idx;
    }

    static std::vector<size_t> parse_renyi_indices(const std::string &renyi_indices_str) {
      std::vector<size_t> indices;
      std::stringstream ss(renyi_indices_str);
      std::string token;

      while (std::getline(ss, token, ',')) {
        try {
          uint32_t number = std::stoi(dataframe::utils::strip(token));
          indices.push_back(number);
        } catch (const std::exception &e) {}
      }

      return indices;
    }
};

