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

    bool sample_leaf_removal;
    size_t num_steps;


  public:
    LinearCodeSampler()=default;

    LinearCodeSampler(dataframe::Params& params) {
      inplace = dataframe::utils::get<int>(params, "inplace", false);
      spacing = dataframe::utils::get<int>(params, "spacing", 1);
      num_steps = dataframe::utils::get<int>(params, "num_leaf_removal_steps", 0);

      sample_rank = dataframe::utils::get<int>(params, "sample_rank", true);
      sample_entanglement = dataframe::utils::get<int>(params, "sample_entanglement", false);
      sample_all_entanglement = dataframe::utils::get<int>(params, "sample_all_entanglement", false);
    }

    void add_entanglement_samples(dataframe::DataSlide &slide, std::shared_ptr<GeneratorMatrix> matrix, const std::vector<size_t>& sites) const {
      slide.add_data("locality");
      slide.push_data("locality", static_cast<double>(matrix->generator_locality(sites)));
    }

    void add_all_entanglement_samples(dataframe::DataSlide& slide, std::shared_ptr<GeneratorMatrix> matrix) const {
      std::vector<dataframe::Sample> s;
      std::vector<size_t> sites;
      size_t num_samples = matrix->num_cols / spacing;
      size_t n = 0;
      for (size_t i = 0; i < num_samples; i++) {
        // Sample locality
        s.push_back(static_cast<double>(matrix->generator_locality(sites)));

        // Add new sites
        for (size_t j = 0; j < spacing; j++) {
          sites.push_back(n);
          n++;
        }
      }

      slide.add_data("locality");
      slide.push_data("locality", s);
    }

    void add_leaf_removal_samples(dataframe::DataSlide& slide, std::shared_ptr<ParityCheckMatrix> matrix, std::minstd_rand& rng) const {
      auto data = matrix->leaf_removal(num_steps, rng);
      slide.add_data("leafs");
      for (size_t i = 0; i < data.size(); i++) {
        // Need to cast to doubles before adding to slide
        std::vector<double> samples(data[i].size());
        std::transform(data[i].begin(), data[i].end(), samples.begin(), [](size_t val) { return static_cast<double>(val); });
        slide.push_data("leafs", samples);
      }
    }

    void add_samples(dataframe::DataSlide &slide, LinearCodeMatrix matrix, std::minstd_rand& rng, const std::optional<std::vector<size_t>>& sites = std::nullopt) {
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
        slide.add_data("rank");
        slide.push_data("rank", H->rank(inplace));
      }

      if (sample_entanglement) {
        std::vector<size_t> sites_val;
        if (!sites.has_value()) {
          throw std::invalid_argument("If sampling partial entanglement, must provide list of sites.");
        }

        add_entanglement_samples(slide, G, sites.value());
      }

      if (sample_all_entanglement) {
        add_all_entanglement_samples(slide, G);
      }

      if (sample_leaf_removal) {
        add_leaf_removal_samples(slide, H, rng);
      }
    }
};
