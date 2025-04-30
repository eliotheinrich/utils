#pragma once

#include "MPSMagicSampler.hpp"

class MPSMagicSampler : public StabilizerEntropySampler {
  public:
    MPSMagicSampler(dataframe::ExperimentParams& params) : StabilizerEntropySampler(params) {}

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<MagicQuantumState>& state_) override {
      std::shared_ptr<MatrixProductState> state = std::dynamic_pointer_cast<MatrixProductState>(state_);

      if (sample_stabilizer_entropy || sample_stabilizer_entropy_mutual || sample_stabilizer_entropy_bipartite) {
        std::vector<QubitSupport> supports;
        if (sample_stabilizer_entropy_mutual) {
          uint32_t nqb = state->get_num_qubits();
          QubitInterval qubitsA = std::make_pair(0, stabilizer_entropy_mutual_subsystem_size);
          supports.push_back(qubitsA);
          QubitInterval qubitsB = std::make_pair(nqb - stabilizer_entropy_mutual_subsystem_size, nqb);
          supports.push_back(qubitsB);
        }
        std::vector<PauliAmplitudes> pauli_samples = state->sample_paulis(supports, sre_num_samples);
        std::vector<std::vector<double>> amplitudes = extract_amplitudes(pauli_samples);

        size_t num_qubits = state_->get_num_qubits();
        if (sample_stabilizer_entropy) {
          std::vector<double> amplitudes_ = normalize_pauli_samples(amplitudes[0], num_qubits, state_->purity());

          std::vector<double> stabilizer_renyi_entropy;
          for (auto index : renyi_indices) {
            double M = estimate_renyi_entropy(index, amplitudes_) - num_qubits * std::log(2);
            dataframe::utils::emplace(samples, STABILIZER_ENTROPY(index), M);
          }
        }

        if (sample_stabilizer_entropy_mutual) {
          double M = MatrixProductState::calculate_magic_mutual_information_from_samples2(amplitudes[0], amplitudes[1], amplitudes[2]);
          dataframe::utils::emplace(samples, STABILIZER_ENTROPY_MUTUAL, M);
        }

        if (sample_stabilizer_entropy_bipartite) {
          std::vector<double> M = state->process_bipartite_pauli_samples(pauli_samples);
          dataframe::utils::emplace(samples, STABILIZER_ENTROPY_BIPARTITE, M);
        }
      }
    }
};

