#pragma once

#include <Frame.h>
#include "QuantumState.h"

#include <iostream>
#include <string>

class MagicStateSampler {
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

    MagicStateSampler(dataframe::ExperimentParams& params) {
      sre_use_parent_methods = dataframe::utils::get<int>(params, "sre_use_parent_methods", false);

      sample_stabilizer_entropy = dataframe::utils::get<int>(params, "sample_stabilizer_entropy", false);
      sre_save_samples = dataframe::utils::get<int>(params, "sre_save_samples", 0);
      sre_num_samples = dataframe::utils::get<int>(params, "sre_num_samples", 1000);
      sre_method = parse_sre_method(dataframe::utils::get<std::string>(params, "sre_method", "virtual"));
      if (sre_method == sre_method_t::MonteCarlo) {
        sre_mc_equilibration_timesteps = dataframe::utils::get<int>(params, "sre_mc_equilibration_timesteps");
      }

      if (sample_stabilizer_entropy) {
        renyi_indices = parse_renyi_indices(dataframe::utils::get<std::string>(params, "stabilizer_entropy_indices", "1"));
      }

      sample_stabilizer_entropy_mutual = dataframe::utils::get<int>(params, "sample_stabilizer_entropy_mutual", false);
      if (sample_stabilizer_entropy_mutual) {
        stabilizer_entropy_mutual_subsystem_size = dataframe::utils::get<int>(params, "stabilizer_entropy_mutual_subsystem_size");
      }

      sample_stabilizer_entropy_bipartite = dataframe::utils::get<int>(params, "sample_stabilizer_entropy_bipartite", false);
      if (sample_stabilizer_entropy_mutual || sample_stabilizer_entropy_bipartite) {
        sre_mmi_save_samples = dataframe::utils::get<int>(params, "sre_mmi_save_samples", false);
      }
    }

    ~MagicStateSampler()=default;

    void set_montecarlo_update(PauliMutationFunc f) {
      mutation = f;
    }

    void add_stabilizer_entropy_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) {
      auto compute_sre_montecarlo = [&](std::shared_ptr<MagicQuantumState> state, const std::vector<size_t>& indices, const std::vector<PauliAmplitudes>& pauli_samples) {
        std::vector<double> amplitudes = extract_amplitudes(pauli_samples)[0];
        double purity = std::pow(2.0, state->get_num_qubits()) * state->purity();
        std::transform(amplitudes.begin(), amplitudes.end(), amplitudes.begin(), [&](auto x) { return x * x / purity; });

        std::vector<double> stabilizer_renyi_entropy;
        for (auto index : indices) {
          double M = estimate_renyi_entropy(index, amplitudes);
          stabilizer_renyi_entropy.push_back(M);
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

      auto compute_sre_exhaustive = [&](std::shared_ptr<MagicQuantumState> state, const std::vector<size_t>& indices, const std::vector<PauliAmplitudes>& pauli_samples) {
        std::vector<double> amplitudes = extract_amplitudes(pauli_samples)[0];
        double purity = std::pow(2.0, state->get_num_qubits()) * state->purity();
        std::transform(amplitudes.begin(), amplitudes.end(), amplitudes.begin(), [&](auto x) { return x * x / purity; });

        std::vector<double> stabilizer_renyi_entropy;
        for (auto index : indices) {
          double M = renyi_entropy(index, amplitudes);
          stabilizer_renyi_entropy.push_back(M);
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

      state->set_use_parent_implementation(sre_use_parent_methods);

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

      if (sre_save_samples) {
        dataframe::utils::emplace(samples, "stabilizer_renyi_entropy_amplitudes", amplitudes);
      }

      for (size_t i = 0; i < renyi_indices.size(); i++) {
        dataframe::utils::emplace(samples, fmt::format("stabilizer_renyi_entropy{}", renyi_indices[i]), stabilizer_renyi_entropy[i]);
      }
    }

    void add_mmi_samples(dataframe::SampleMap& samples, const std::vector<MutualMagicData>& data) const {
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

    void add_stabilizer_entropy_mutual_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) {
      size_t num_qubits = state->get_num_qubits();
      if (stabilizer_entropy_mutual_subsystem_size > num_qubits/2) {
        throw std::runtime_error(fmt::format("stabilizer_entropy_mutual_subsystem_size = {}, but num_qubits = {}", stabilizer_entropy_mutual_subsystem_size, num_qubits));
      }

      std::vector<uint32_t> qubitsA(stabilizer_entropy_mutual_subsystem_size);
      std::vector<uint32_t> qubitsB(stabilizer_entropy_mutual_subsystem_size);
      for (size_t i = 0; i < stabilizer_entropy_mutual_subsystem_size; i++) {
        qubitsA[i] = i;
        qubitsB[i] = i + num_qubits - stabilizer_entropy_mutual_subsystem_size;
      }

      double mmi_sample;
      if (sre_method == sre_method_t::Exhaustive) {
        mmi_sample = state->magic_mutual_information_exhaustive(qubitsA, qubitsB);
      } else if (sre_method == sre_method_t::MonteCarlo) {
        if (sre_mmi_save_samples) {
          auto mmi_data = state->magic_mutual_information_samples_montecarlo(qubitsA, qubitsB, sre_num_samples, sre_mc_equilibration_timesteps, mutation);
          add_mmi_samples(samples, {mmi_data});
          mmi_sample = MagicQuantumState::calculate_magic_mutual_information_from_samples(mmi_data);
        } else {
          mmi_sample = state->magic_mutual_information_montecarlo(qubitsA, qubitsB, sre_num_samples, sre_mc_equilibration_timesteps, mutation);
        }
      } else if (sre_method == sre_method_t::Exact) {
        if (sre_mmi_save_samples) {
          auto mmi_data = state->magic_mutual_information_samples_exact(qubitsA, qubitsB, sre_num_samples);
          add_mmi_samples(samples, {mmi_data});
          mmi_sample = MagicQuantumState::calculate_magic_mutual_information_from_samples(mmi_data);
        } else {
          mmi_sample = state->magic_mutual_information_exact(qubitsA, qubitsB, sre_num_samples);
        }
      } else if (sre_method == sre_method_t::Virtual) {
        mmi_sample = state->magic_mutual_information(qubitsA, qubitsB, sre_num_samples);
      }

      dataframe::utils::emplace(samples, "magic_mutual_information", mmi_sample);
    }
    
    void add_stabilizer_entropy_bipartite_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) {
      std::vector<double> mmi_samples;

      if (sre_method == sre_method_t::Exhaustive) {
        mmi_samples = state->bipartite_magic_mutual_information_exhaustive();
      } else if (sre_method == sre_method_t::MonteCarlo) {
        if (sre_mmi_save_samples) {
          auto mmi_data = state->bipartite_magic_mutual_information_samples_montecarlo(sre_num_samples, sre_mc_equilibration_timesteps, mutation);
          add_mmi_samples(samples, mmi_data);
          mmi_samples.resize(mmi_data.size());
          std::transform(mmi_data.begin(), mmi_data.end(), mmi_samples.begin(), [](const MutualMagicData& s) { return MagicQuantumState::calculate_magic_mutual_information_from_samples(s); });
        } else {
          mmi_samples = state->bipartite_magic_mutual_information_montecarlo(sre_num_samples, sre_mc_equilibration_timesteps, mutation);
        }
      } else if (sre_method == sre_method_t::Exact) {
        if (sre_mmi_save_samples) {
          auto mmi_data = state->bipartite_magic_mutual_information_samples_exact(sre_num_samples);
          add_mmi_samples(samples, mmi_data);
          mmi_samples.resize(mmi_data.size());
          std::transform(mmi_data.begin(), mmi_data.end(), mmi_samples.begin(), [](const MutualMagicData& s) { return MagicQuantumState::calculate_magic_mutual_information_from_samples(s); });
        } else {
          mmi_samples = state->bipartite_magic_mutual_information_exact(sre_num_samples);
        }
      } else if (sre_method == sre_method_t::Virtual) {
        mmi_samples = state->bipartite_magic_mutual_information(sre_num_samples);
      }

      dataframe::utils::emplace(samples, "bipartite_magic_mutual_information", mmi_samples);
    }

    void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) {
      if (sample_stabilizer_entropy) {
        add_stabilizer_entropy_samples(samples, state);
      }

      if (sample_stabilizer_entropy_mutual) {
        add_stabilizer_entropy_mutual_samples(samples, state);
      }

      if (sample_stabilizer_entropy_bipartite) {
        add_stabilizer_entropy_bipartite_samples(samples, state);
      }
    }

  private:	
    bool sre_use_parent_methods;

    bool sample_stabilizer_entropy;
    std::vector<size_t> renyi_indices;

    sre_method_t sre_method;
    bool sre_save_samples;
    size_t sre_num_samples;
    size_t sre_mc_equilibration_timesteps;

    bool sample_stabilizer_entropy_mutual;
    bool sre_mmi_save_samples;
    size_t stabilizer_entropy_mutual_subsystem_size;
    bool sample_stabilizer_entropy_bipartite;

    std::optional<PauliMutationFunc> mutation;

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

