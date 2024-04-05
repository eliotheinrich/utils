#pragma once

#include <Frame.h>
#include "BinaryPolynomial.h"

class LinearCodeSampler {
  private:	
    using LinearCodeMatrix = std::variant<std::shared_ptr<GeneratorMatrix>, std::shared_ptr<ParityCheckMatrix>>;
    bool inplace;

    bool sample_rank;

    bool sample_entanglement;
    bool sample_all_entanglement;

    size_t spacing;


  public:
    LinearCodeSampler()=default;

    LinearCodeSampler(dataframe::Params& params) {
      inplace = dataframe::utils::get<int>(params, "inplace", false);
      spacing = dataframe::utils::get<int>(params, "spacing", 1);

      sample_rank = dataframe::utils::get<int>(params, "sample_rank", true);
      sample_entanglement = dataframe::utils::get<int>(params, "sample_entanglement", false);
      sample_all_entanglement = dataframe::utils::get<int>(params, "sample_all_entanglement", false);
    }

    void add_entanglement_samples(dataframe::data_t &samples, std::shared_ptr<GeneratorMatrix> matrix, const std::vector<size_t>& sites) const {
      auto locality = matrix->generator_locality_samples(sites);
      for (size_t a = 0; a < locality.size(); a++) {
        samples.emplace("locality" + std::to_string(a), static_cast<double>(locality[a]));
      }
      //samples.emplace("entanglement", static_cast<double>(matrix->generator_locality(sites)));
    }

    void add_all_entanglement_samples(dataframe::data_t& samples, std::shared_ptr<GeneratorMatrix> matrix) const {
      std::vector<dataframe::Sample> s;
      std::vector<size_t> sites;
      size_t num_samples = matrix->num_cols / spacing;
      size_t n = 0;
      for (size_t i = 0; i < num_samples; i++) {
        // Sample locality
        auto locality = matrix->generator_locality_samples(sites);
        for (size_t a = 0; a < locality.size(); a++) {
          samples.emplace("locality" + std::to_string(a), static_cast<double>(locality[a]));
        }
        //s.push_back(static_cast<double>(matrix->generator_locality(sites)));

        // Add new sites
        for (size_t j = 0; j < spacing; j++) {
          sites.push_back(n);
          n++;
        }
      }

      samples.emplace("entanglement", s);
    }

    void add_samples(dataframe::data_t &samples, LinearCodeMatrix matrix, const std::optional<std::vector<size_t>>& sites = std::nullopt) {
      std::shared_ptr<GeneratorMatrix> G;
      std::shared_ptr<ParityCheckMatrix> H;
      if (matrix.index() == 0) {
        G = std::get<std::shared_ptr<GeneratorMatrix>>(matrix);
        H = std::shared_ptr<ParityCheckMatrix>(new ParityCheckMatrix(G->to_parity_check_matrix()));
      } else {
        H = std::get<std::shared_ptr<ParityCheckMatrix>>(matrix);
        G = std::shared_ptr<GeneratorMatrix>(new GeneratorMatrix(H->to_generator_matrix()));
      }


      if (sample_rank) {
        samples.emplace("rank", H->rank(inplace));
      }

      if (sample_entanglement) {
        std::vector<size_t> sites_val;
        if (!sites.has_value()) {
          throw std::invalid_argument("If sampling partial entanglement, must provide list of sites.");
        }

        add_entanglement_samples(samples, G, sites.value());
      }

      if (sample_all_entanglement) {
        add_all_entanglement_samples(samples, G);
      }
    }
};
