#pragma once 

#include "Frame.h"
#include <QuantumState.h>

#define PARTICIPATION_ENTROPY           "participation_entropy"
#define PARTICIPATION_ENTROPY_MUTUAL    "participation_entropy_mutual"
#define PARTICIPATION_ENTROPY_BIPARTITE "participation_entropy_bipartite"

class ParticipationSampler {
  public:
    ParticipationSampler(dataframe::ExperimentParams& params) {
      sample_participation_entropy = dataframe::utils::get<int>(params, "sample_participation_entropy", false);

      num_participation_entropy_samples = dataframe::utils::get<int>(params, "num_participation_entropy_samples", 1000);

      sample_participation_entropy_mutual = dataframe::utils::get<int>(params, "sample_participation_entropy_mutual", false);

      sample_participation_entropy_bipartite = dataframe::utils::get<int>(params, "sample_participation_entropy_bipartite", false);
    }

    virtual void add_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state)=0;

  protected:
    bool sample_participation_entropy;

    size_t num_participation_entropy_samples;

    bool sample_participation_entropy_mutual;

    bool sample_participation_entropy_bipartite;
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
      double W;
      if (participation_entropy_method == "sampled") {
        auto samples = extract_amplitudes(state->sample_bitstrings({}, num_participation_entropy_samples))[0];
        W = estimate_renyi_entropy(1, samples, 2);
      } else if (participation_entropy_method == "exhaustive") {
        auto probs = state->probabilities();
        W = renyi_entropy(1, probs, 2);
      }

      dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY, W);
    }

    void add_participation_entropy_mutual_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      size_t nqb = state->get_num_qubits();

      Qubits qubitsA(nqb/2);
      std::iota(qubitsA.begin(), qubitsA.end(), 0);

      Qubits qubitsB(nqb/2);
      std::iota(qubitsB.begin(), qubitsB.end(), nqb/2);

      double M;
      if (participation_entropy_method == "sampled") {
        auto stateA = state->partial_trace(qubitsB);
        auto stateB = state->partial_trace(qubitsA);

        auto samplesAB = extract_amplitudes(state->sample_bitstrings({}, num_participation_entropy_samples))[0];
        auto samplesA = extract_amplitudes(stateA->sample_bitstrings({}, num_participation_entropy_samples))[0];
        auto samplesB = extract_amplitudes(stateB->sample_bitstrings({}, num_participation_entropy_samples))[0];

        M = estimate_renyi_entropy(1, samplesA, 2) + estimate_renyi_entropy(1, samplesB, 2) - estimate_renyi_entropy(1, samplesAB, 2);
      } else if (participation_entropy_method == "exhaustive") {
        auto partial_distributions = state->partial_probabilities({qubitsA, qubitsB});
        M = renyi_entropy(1, partial_distributions[1], 2) + renyi_entropy(1, partial_distributions[2], 2) - renyi_entropy(1, partial_distributions[0], 2);
      }

      dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_MUTUAL, M);
    }

    void add_participation_entropy_bipartite_samples(dataframe::SampleMap& samples, const std::shared_ptr<QuantumState>& state) const {
      size_t num_qubits = state->get_num_qubits();
      size_t N = num_qubits/2 - 1;
      std::vector<double> entropy(N);

      auto supports = get_bipartite_supports(num_qubits);
      if (participation_entropy_method == "sampled") {
        auto samples = extract_amplitudes(state->sample_bitstrings(supports, num_participation_entropy_samples));

        double W = estimate_renyi_entropy(1, samples[0], 2);
        for (size_t i = 0; i < N; i++) {
          double WA = estimate_renyi_entropy(1, samples[i + 1], 2);
          double WB = estimate_renyi_entropy(1, samples[i + 1 + N], 2);
          entropy[i] = WA + WB - W;
        }
      } else if (participation_entropy_method == "exhaustive") {
        std::vector<double> entropy_(N);

        auto partials = state->partial_probabilities(supports);
        double W = renyi_entropy(1, partials[0], 2);

        auto supports = get_bipartite_supports(num_qubits);
        for (size_t i = 0; i < N; i++) {
          double WA = renyi_entropy(1, partials[i + 1], 2);
          double WB = renyi_entropy(1, partials[i + 1 + N], 2);

          entropy[i] = WA + WB - W;
        }
      }

      dataframe::utils::emplace(samples, PARTICIPATION_ENTROPY_BIPARTITE, entropy);
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
