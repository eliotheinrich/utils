#pragma once

#include "FreeFermion.h"
#include <Frame.h>

class FreeFermionSampler {
  private:
    bool sample_correlations;

  public:
    FreeFermionSampler(dataframe::ExperimentParams& params) {
      sample_correlations = dataframe::utils::get<int>(params, "sample_correlations", 0);
    }

    void add_correlation_samples(dataframe::SampleMap& samples, std::shared_ptr<GaussianState>& state) const {
      auto C = state->covariance_matrix();
      size_t L = C.size() / 2;
      std::vector<std::vector<double>> correlations(L, std::vector<double>(L));

      for (size_t r = 0; r < L; r++) {
        // Average over space
        for (size_t i = 0; i < L; i++) {
          double c = std::abs(C(i, (i+r)%L));
          correlations[r][i] = c*c;
        }
      }

      std::vector<size_t> shape = {L};
      auto data = dataframe::utils::samples_to_dataobject(correlations, shape);
      dataframe::utils::emplace(samples, "correlations", std::move(data));
    }

    void take_samples(dataframe::SampleMap& samples, std::shared_ptr<GaussianState>& state) {
      if (sample_correlations) {
        add_correlation_samples(samples, state);
      }

      dataframe::utils::emplace(samples, "num_particles", state->num_particles());
      dataframe::utils::emplace(samples, "num_real_particles", state->num_real_particles());
    }
};
