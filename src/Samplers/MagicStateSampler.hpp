#pragma once

#include <Frame.h>
#include "QuantumState.h"

#include <iostream>
#include <string>

#define STABILIZER_ENTROPY(x) fmt::format("stabilizer_entropy{}", x)
#define STABILIZER_ENTROPY_MUTUAL "stabilizer_entropy_mutual"
#define STABILIZER_ENTROPY_BIPARTITE "stabilizer_entropy_bipartite"

class StabilizerEntropySampler {
  public:
    StabilizerEntropySampler(dataframe::ExperimentParams& params) {
      sample_stabilizer_entropy = dataframe::utils::get<int>(params, "sample_stabilizer_entropy", false);
      sre_num_samples = dataframe::utils::get<int>(params, "sre_num_samples", 1000);

      if (sample_stabilizer_entropy) {
        renyi_indices = parse_renyi_indices(dataframe::utils::get<std::string>(params, "stabilizer_entropy_indices", "1"));
      }

      sample_stabilizer_entropy_mutual = dataframe::utils::get<int>(params, "sample_stabilizer_entropy_mutual", false);
      if (sample_stabilizer_entropy_mutual) {
        stabilizer_entropy_mutual_subsystem_size = dataframe::utils::get<int>(params, "stabilizer_entropy_mutual_subsystem_size");
      }

      sample_stabilizer_entropy_bipartite = dataframe::utils::get<int>(params, "sample_stabilizer_entropy_bipartite", false);
    }

    ~StabilizerEntropySampler()=default;

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state)=0;

  protected:	
    bool sre_use_parent_methods;

    bool sample_stabilizer_entropy;
    std::vector<size_t> renyi_indices;

    size_t sre_num_samples;

    bool sample_stabilizer_entropy_mutual;
    size_t stabilizer_entropy_mutual_subsystem_size;
    bool sample_stabilizer_entropy_bipartite;

  private:
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

class MagicStateSampler : public StabilizerEntropySampler {
  public:
    enum sre_method_t {
      MonteCarlo, Exhaustive, Exact
    };

    sre_method_t parse_sre_method(const std::string& s) {
      if (s == "montecarlo") {
        return sre_method_t::MonteCarlo;
      } else if (s == "exhaustive") {
        return sre_method_t::Exhaustive;
      } else if (s == "exact") {
        return sre_method_t::Exact;
      } else {
        throw std::runtime_error(fmt::format("Stabilizer renyi entropy method \"{}\" not found.", s));
      }
    }

    MagicStateSampler(dataframe::ExperimentParams& params) : StabilizerEntropySampler(params) {
      sre_method = parse_sre_method(dataframe::utils::get<std::string>(params, "sre_method"));
      if (sre_method == sre_method_t::MonteCarlo) {
        sre_mc_equilibration_timesteps = dataframe::utils::get<int>(params, "sre_mc_equilibration_timesteps");
      }

      if (sample_stabilizer_entropy) {
        sre_save_samples = dataframe::utils::get<int>(params, "sre_save_samples", false);
      }

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
        std::vector<double> amplitudes = normalize_pauli_samples(extract_amplitudes(pauli_samples)[0], state->get_num_qubits(), state->purity());

        size_t num_qubits = state->get_num_qubits();
        std::vector<double> stabilizer_renyi_entropy;
        for (auto index : indices) {
          double M = estimate_renyi_entropy(index, amplitudes);
          stabilizer_renyi_entropy.push_back(M - num_qubits * std::log(2));
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

      auto compute_sre_exhaustive = [&](std::shared_ptr<MagicQuantumState> state, const std::vector<size_t>& indices, const std::vector<PauliAmplitudes>& pauli_samples) {
        std::vector<double> amplitudes = extract_amplitudes(pauli_samples)[0];
        double N = 0.0;
        for (size_t i = 0; i < amplitudes.size(); i++) {
          N += amplitudes[i] * amplitudes[i];
        }

        for (size_t i = 0; i < amplitudes.size(); i++) {
          amplitudes[i] = amplitudes[i] * amplitudes[i] / N;
        }

        size_t num_qubits = state->get_num_qubits();
        std::vector<double> stabilizer_renyi_entropy;
        for (auto index : indices) {
          double M = renyi_entropy(index, amplitudes);
          stabilizer_renyi_entropy.push_back(M - num_qubits * std::log(2));
        }
        return std::make_pair(amplitudes, stabilizer_renyi_entropy);
      };

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
      }

      if (sre_save_samples) {
        dataframe::utils::emplace(samples, "stabilizer_entropy_amplitudes", amplitudes);
      }

      for (size_t i = 0; i < renyi_indices.size(); i++) {
        dataframe::utils::emplace(samples, STABILIZER_ENTROPY(renyi_indices[i]), stabilizer_renyi_entropy[i]);
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
        auto tAB2_ = t2[0];
        auto tA2_ = t2[1];
        auto tB2_ = t2[2];

        auto tAB4_ = t4[0];
        auto tA4_ = t4[1];
        auto tB4_ = t4[2];

        tA2.push_back(tA2_);
        tB2.push_back(tB2_);
        tAB2.push_back(tAB2_);
        tA4.push_back(tA4_);
        tB4.push_back(tB4_);
        tAB4.push_back(tAB4_);
      }

      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tA2", tA2);
      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tB2", tB2);
      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tAB2", tAB2);
      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tA4", tA4);
      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tB4", tB4);
      dataframe::utils::emplace(samples, "stabilizer_entropy_mutual_tAB4", tAB4);
    }

    void add_stabilizer_entropy_mutual_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) {
      size_t num_qubits = state->get_num_qubits();
      if (stabilizer_entropy_mutual_subsystem_size > num_qubits/2) {
        throw std::runtime_error(fmt::format("stabilizer_entropy_mutual_subsystem_size = {}, but num_qubits = {}", stabilizer_entropy_mutual_subsystem_size, num_qubits));
      }

      Qubits qubitsA = to_qubits(std::make_pair(0, stabilizer_entropy_mutual_subsystem_size));
      Qubits qubitsB = to_qubits(std::make_pair(num_qubits - stabilizer_entropy_mutual_subsystem_size, num_qubits));

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
      }

      dataframe::utils::emplace(samples, STABILIZER_ENTROPY_MUTUAL, mmi_sample);
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
      }

      dataframe::utils::emplace(samples, STABILIZER_ENTROPY_BIPARTITE, mmi_samples);
    }

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state) override {
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
    sre_method_t sre_method;
    size_t sre_mc_equilibration_timesteps;

    bool sre_save_samples;
    bool sre_mmi_save_samples;

    std::optional<PauliMutationFunc> mutation;
};

