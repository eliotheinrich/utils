#pragma once 

#include "Frame.h"
#include "StabilizerEntropySampler.hpp" // for parse_renyi_indices
#include <QuantumState.h>
#include <CliffordState.h>

#define PARTICIPATION_ENTROPY(x) fmt::format("participation_entropy{}", x)
#define PARTICIPATION_ENTROPY_MUTUAL(x) fmt::format("participation_entropy_mutual{}", x)
#define PARTICIPATION_ENTROPY_BIPARTITE(x) fmt::format("participation_entropy_bipartite{}", x)

class ParticipationSampler {
  public:
    ParticipationSampler(dataframe::ExperimentParams& params) {
      sample_participation_entropy = dataframe::utils::get<int>(params, "sample_participation_entropy", false);

      num_participation_entropy_samples = dataframe::utils::get<int>(params, "num_participation_entropy_samples", 1000);

      sample_participation_entropy_mutual = dataframe::utils::get<int>(params, "sample_participation_entropy_mutual", false);

      sample_participation_entropy_bipartite = dataframe::utils::get<int>(params, "sample_participation_entropy_bipartite", false);

      renyi_indices = parse_renyi_indices(dataframe::utils::get<std::string>(params, "participation_entropy_indices", "1"));
    }

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state)=0;

  protected:
    bool sample_participation_entropy;

    size_t num_participation_entropy_samples;

    bool sample_participation_entropy_mutual;

    bool sample_participation_entropy_bipartite;

    std::vector<size_t> renyi_indices;
};

class GenericParticipationSampler : public ParticipationSampler {
  public:
    GenericParticipationSampler(dataframe::ExperimentParams& params) : ParticipationSampler(params) {
      participation_entropy_method = dataframe::utils::get<std::string>(params, "participation_entropy_method", "sampled");
      std::set<std::string> allowed_methods = {"sampled", "exhaustive"};
      if (!allowed_methods.contains(participation_entropy_method)) {
        throw std::runtime_error(fmt::format("Invalid participation entropy method: {}\n", participation_entropy_method));
      }
    }

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) override {
      if (sample_participation_entropy) {
        add_participation_entropy_samples(samples, state);
      }

      if (sample_participation_entropy_mutual) {
        add_participation_entropy_mutual_samples(samples, state);
      }

      if (sample_participation_entropy_bipartite) {
        add_participation_entropy_bipartite_samples(samples, state);
      }
    }

  private:
    std::string participation_entropy_method;

    void add_participation_entropy_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      std::vector<double> W(renyi_indices.size());
      if (participation_entropy_method == "sampled") {
        auto prob_samples = extract_amplitudes(state->sample_bitstrings({}, num_participation_entropy_samples))[0];
        for (size_t i = 0; i < renyi_indices.size(); i++) {
          W[i] = estimate_renyi_entropy(renyi_indices[i], prob_samples, 2);
        }
      } else if (participation_entropy_method == "exhaustive") {
        auto probs = state->probabilities();
        for (size_t i = 0; i < renyi_indices.size(); i++) {
          W[i] = renyi_entropy(renyi_indices[i], probs, 2);
        }
      }

      for (size_t i = 0; i < renyi_indices.size(); i++) {
        dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY(renyi_indices[i]), W[i]);
      }
    }

    void add_participation_entropy_mutual_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      size_t nqb = state->get_num_qubits();

      Qubits qubitsA(nqb/2);
      std::iota(qubitsA.begin(), qubitsA.end(), 0);

      Qubits qubitsB(nqb/2);
      std::iota(qubitsB.begin(), qubitsB.end(), nqb/2);

      std::vector<double> M(renyi_indices.size());
      if (participation_entropy_method == "sampled") {
        auto stateA = state->partial_trace(qubitsB);
        auto stateB = state->partial_trace(qubitsA);

        auto samplesAB = extract_amplitudes(state->sample_bitstrings({}, num_participation_entropy_samples))[0];
        auto samplesA = extract_amplitudes(stateA->sample_bitstrings({}, num_participation_entropy_samples))[0];
        auto samplesB = extract_amplitudes(stateB->sample_bitstrings({}, num_participation_entropy_samples))[0];

        for (size_t i = 0; i < renyi_indices.size(); i++) {
          size_t idx = renyi_indices[i];
          M[i] = estimate_renyi_entropy(idx, samplesA, 2) + estimate_renyi_entropy(idx, samplesB, 2) - estimate_renyi_entropy(idx, samplesAB, 2);
        }
      } else if (participation_entropy_method == "exhaustive") {
        auto partial_distributions = state->partial_probabilities({qubitsA, qubitsB});
        for (size_t i = 0; i < renyi_indices.size(); i++) {
          size_t idx = renyi_indices[i];
          M[i] = renyi_entropy(idx, partial_distributions[1], 2) + renyi_entropy(idx, partial_distributions[2], 2) - renyi_entropy(idx, partial_distributions[0], 2);
        }
      }

      for (size_t i = 0; i < renyi_indices.size(); i++) {
        dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_MUTUAL(renyi_indices[i]), M[i]);
      }
    }

    void add_participation_entropy_bipartite_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      size_t num_qubits = state->get_num_qubits();
      size_t N = num_qubits/2 - 1;
      std::vector<std::vector<double>> entropy(renyi_indices.size(), std::vector<double>(N));

      auto supports = get_bipartite_supports(num_qubits);
      if (participation_entropy_method == "sampled") {
        auto samples = extract_amplitudes(state->sample_bitstrings(supports, num_participation_entropy_samples));

        for (size_t i = 0; i < renyi_indices.size(); i++) {
          size_t idx = renyi_indices[i];
          double W = estimate_renyi_entropy(idx, samples[0], 2);
          for (size_t j = 0; j < N; j++) {
            double WA = estimate_renyi_entropy(idx, samples[j + 1], 2);
            double WB = estimate_renyi_entropy(idx, samples[j + 1 + N], 2);
            entropy[i][j] = WA + WB - W;
          }
        }
      } else if (participation_entropy_method == "exhaustive") {
        std::vector<double> entropy_(N);

        auto partials = state->partial_probabilities(supports);
        auto supports = get_bipartite_supports(num_qubits);

        for (size_t i = 0; i < renyi_indices.size(); i++) {
          size_t idx = renyi_indices[i];
          double W = renyi_entropy(idx, partials[0], 2);
          for (size_t j = 0; j < N; j++) {
            double WA = renyi_entropy(idx, partials[j + 1], 2);
            double WB = renyi_entropy(idx, partials[j + 1 + N], 2);

            entropy[i][j] = WA + WB - W;
          }
        }
      }

      for (size_t i = 0; i < renyi_indices.size(); i++) {
        dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_BIPARTITE(renyi_indices[i]), entropy[i]);
      }
    }
};

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
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            double W = estimate_renyi_entropy(renyi_indices[i], amplitudes[0], 2);
            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY(renyi_indices[i]), W);
          }
        }

        if (sample_participation_entropy_mutual) {
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            double L = estimate_mutual_renyi_entropy(renyi_indices[i], amplitudes[0], amplitudes[1], amplitudes[2], 2);
            dataframe::utils::emplace(samples, "mutual1", estimate_renyi_entropy(renyi_indices[i], amplitudes[0], 2));
            dataframe::utils::emplace(samples, "mutual2", estimate_renyi_entropy(renyi_indices[i], amplitudes[1], 2));
            dataframe::utils::emplace(samples, "mutual3", estimate_renyi_entropy(renyi_indices[i], amplitudes[2], 2));
            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_MUTUAL(renyi_indices[i]), L);
          }
        }

        if (sample_participation_entropy_bipartite) {
          std::vector<std::vector<double>> L = state->process_bipartite_bit_samples(renyi_indices, bit_samples);
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_BIPARTITE(renyi_indices[i]), L[i]);
          }
        }
      }
    }
};

class CHPParticipationSampler : public ParticipationSampler {
  public:
    CHPParticipationSampler(dataframe::ExperimentParams& params) : ParticipationSampler(params) {}

    ~CHPParticipationSampler()=default;

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state_) override {
      std::shared_ptr<QuantumCHPState> state = std::dynamic_pointer_cast<QuantumCHPState>(state_);

      if (sample_participation_entropy || sample_participation_entropy_mutual || sample_participation_entropy_bipartite) {
        // Only perform sample a single time
        double pe = state->xrank();
        uint32_t num_qubits = state->get_num_qubits();

        if (sample_participation_entropy) {
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            if (renyi_indices[i] != 2) {
              throw std::runtime_error("Have not yet implemented n != 2 index for participation entropy for CHP states.");
            }

            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY(renyi_indices[i]), pe);
          }
        }

        if (sample_participation_entropy_mutual) {
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            if (renyi_indices[i] != 2) {
              throw std::runtime_error("Have not yet implemented n != 2 index for participation entropy for CHP states.");
            }

            Qubits A = to_qubits(std::make_pair(0, num_qubits/2));
            Qubits B = to_qubits(std::make_pair(num_qubits/2, num_qubits));
            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_MUTUAL(renyi_indices[i]), state->partial_xrank(A) + state->partial_xrank(B) - pe);
          }
        }

        if (sample_participation_entropy_bipartite) {
          for (size_t i = 0; i < renyi_indices.size(); i++) {
            if (renyi_indices[i] != 2) {
              throw std::runtime_error("Have not yet implemented n != 2 index for participation entropy for CHP states.");
            }

            size_t N = num_qubits/2 - 1;
            auto supports = get_bipartite_supports(num_qubits);
            std::vector<double> W(N);
            for (size_t j = 0; j < N; j++) {
              Qubits A = to_qubits(supports[j]);
              Qubits B = to_qubits(supports[j + N]);

              W[j] = state->partial_xrank(A) + state->partial_xrank(B) - state->xrank();
            }
            dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_BIPARTITE(renyi_indices[i]), W);
          }
        }
      }
    }
};
