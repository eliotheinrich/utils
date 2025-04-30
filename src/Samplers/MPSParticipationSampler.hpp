#pragma once

#include "QuantumStateSampler.hpp"

class MPSParticipationSampler : public ParticipationSampler {
  public:
    MPSParticipationSampler(dataframe::ExperimentParams& params) : ParticipationSampler(params) {}

    ~MPSParticipationSampler()=default;

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state_) override {
      std::shared_ptr<MatrixProductState> state = std::dynamic_pointer_cast<MatrixProductState>(state_);

      if (sample_participation_entropy || sample_participation_entropy_mutual || sample_participation_entropy_bipartite) {
        // Only perform sample a single time
        std::vector<QubitSupport> supports;
        if (sample_participation_entropy_mutual) {
          uint32_t nqb = state->get_num_qubits();
          QubitInterval qubitsA = std::make_pair(0, nqb/2);
          supports.push_back(qubitsA);
          QubitInterval qubitsB = std::make_pair(nqb/2, nqb);
          supports.push_back(qubitsB);
        }
        auto bit_samples = state->sample_bitstrings(supports, num_participation_entropy_samples);
        std::vector<std::vector<double>> amplitudes = extract_amplitudes(bit_samples);

        if (sample_participation_entropy) {
          double W = estimate_renyi_entropy(1, amplitudes[0], 2);
          dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY, W);
        }

        if (sample_participation_entropy_mutual) {
          double L = estimate_mutual_renyi_entropy(amplitudes[0], amplitudes[1], amplitudes[2], 2);
          dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_MUTUAL, L);
        }

        if (sample_participation_entropy_bipartite) {
          std::vector<double> L = state->process_bipartite_bit_samples(bit_samples);
          dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_BIPARTITE, L);
        }
      }
    }
};

