#pragma once

#include <Frame.h>
#include "BinaryPolynomial.h"

class LinearCodeSampler {
  private:	
    using LinearCodeMatrix = std::variant<std::shared_ptr<GeneratorMatrix>, std::shared_ptr<ParityCheckMatrix>>;
    bool inplace;

    bool sample_rank;
    bool sample_solveable;

    bool sample_sym;

    bool sample_locality;

    bool sample_all_locality;
    size_t spacing;

    bool sample_leaf_removal;
    size_t num_steps;
    size_t max_size;
    bool include_isolated_in_core;


  public:
    LinearCodeSampler()=default;

    LinearCodeSampler(dataframe::Params& params) {
      inplace = dataframe::utils::get<int>(params, "inplace", false);

      sample_rank = dataframe::utils::get<int>(params, "sample_rank", true);
      sample_solveable = dataframe::utils::get<int>(params, "sample_solveable", true);
      sample_locality = dataframe::utils::get<int>(params, "sample_locality", false);

      sample_all_locality = dataframe::utils::get<int>(params, "sample_all_locality", false);
      spacing = dataframe::utils::get<int>(params, "spacing", 1);

      sample_sym = dataframe::utils::get<int>(params, "sample_sym", false);

      sample_leaf_removal = dataframe::utils::get<int>(params, "sample_leaf_removal", false);
      num_steps = dataframe::utils::get<int>(params, "num_leaf_removal_steps", 0);
      max_size = dataframe::utils::get<int>(params, "max_size", 0);
      include_isolated_in_core = dataframe::utils::get<int>(params, "include_isolated_in_core", false);
    }

    void add_sym_samples(dataframe::DataSlide &slide, std::shared_ptr<GeneratorMatrix> matrix, const std::vector<size_t>& sites1, const std::vector<size_t>& sites2) const {
      std::vector<size_t> all_sites;
      all_sites.insert(all_sites.end(), sites1.begin(), sites1.end());
      all_sites.insert(all_sites.end(), sites2.begin(), sites2.end());

      slide.add_data("sym");
      slide.push_samples_to_data("sym", (double) matrix->partial_rank(sites1) + matrix->partial_rank(sites2) - matrix->partial_rank(all_sites));
    }

    void add_locality_samples(dataframe::DataSlide &slide, std::shared_ptr<GeneratorMatrix> matrix, const std::vector<size_t>& sites) const {
      slide.add_data("locality");
      slide.push_samples_to_data("locality", static_cast<double>(matrix->generator_locality(sites)));
    }

    void add_all_locality_samples(dataframe::DataSlide& slide, std::shared_ptr<GeneratorMatrix> matrix) const {
      std::vector<size_t> sites;
      size_t num_samples = matrix->num_cols / spacing;
      std::vector<double> s(num_samples);
      size_t n = 0;
      for (size_t i = 0; i < num_samples; i++) {
        // Sample locality
        s[i] = static_cast<double>(matrix->generator_locality(sites));

        // Add new sites
        for (size_t j = 0; j < spacing; j++) {
          sites.push_back(n);
          n++;
        }
      }

      slide.add_data("locality", num_samples);
      slide.push_samples_to_data("locality", s);
    }

    void add_leaf_removal_samples(dataframe::DataSlide& slide, std::shared_ptr<ParityCheckMatrix> matrix, std::minstd_rand& rng) const {
      ParityCheckMatrix H(*matrix.get());
      size_t _num_steps = num_steps ? num_steps : H.num_rows;
      size_t _max_size = max_size ? max_size : H.num_rows;
      std::vector<std::vector<size_t>> sizes(_num_steps);

      std::optional<size_t> r = 0;
      std::vector<size_t> s;
      size_t n = 0;
      while (r.has_value() && n < _num_steps) {
        std::tie(r, s) = H.leaf_removal_iteration(rng);
        s.resize(_max_size, 0u);
        sizes[n] = s;

        n++;
      }


      for (size_t i = n; i < _num_steps; i++) {
        sizes[i] = sizes[n-1];
      }

      if (!include_isolated_in_core) {
        H.reduce();
      }

      size_t core_size = H.num_rows;
      slide.add_data("core_size");
      slide.push_samples_to_data("core_size", core_size);

      slide.add_data("leafs", _max_size);
      for (size_t i = 0; i < _num_steps; i++) {
        std::vector<double> samples(_max_size);
        std::transform(sizes[i].begin(), sizes[i].end(), samples.begin(), [](size_t val) { return static_cast<double>(val); });
        slide.push_samples_to_data("leafs", samples);
      }
    }

    void add_samples(dataframe::DataSlide &slide, LinearCodeMatrix matrix, std::minstd_rand& rng, const std::optional<std::vector<size_t>>& sites1 = std::nullopt, const std::optional<std::vector<size_t>>& sites2 = std::nullopt) {
      std::shared_ptr<GeneratorMatrix> G;
      std::shared_ptr<ParityCheckMatrix> H;
      if (matrix.index() == 0) {
        G = std::get<std::shared_ptr<GeneratorMatrix>>(matrix);
        H = std::shared_ptr<ParityCheckMatrix>(new ParityCheckMatrix(G->to_parity_check_matrix()));
      } else {
        H = std::get<std::shared_ptr<ParityCheckMatrix>>(matrix);
        G = std::shared_ptr<GeneratorMatrix>(new GeneratorMatrix(H->to_generator_matrix()));
      }


      uint32_t r;
      if (sample_rank) {
        slide.add_data("rank");
        r = H->rank(inplace);
        slide.push_samples_to_data("rank", r);
      }

      if (sample_solveable) {
        if (!sample_rank) {
          r = H->rank(inplace);
        }

        bool solveable = (r >= H->num_rows);
        slide.add_data("solveable");
        slide.push_samples_to_data("solveable", solveable);
      }

      if (sample_locality) {
        if (!sites1.has_value()) {
          throw std::invalid_argument("If sampling partial locality, must provide list of sites.");
        }

        add_locality_samples(slide, G, sites1.value());
      }

      if (sample_sym) {
        if (!sites1.has_value() || !sites2.has_value()) {
          throw std::invalid_argument("If sampling sym, must provide list of sites.");
        }

        add_sym_samples(slide, G, sites1.value(), sites2.value());
      }

      if (sample_all_locality) {
        add_all_locality_samples(slide, G);
      }

      if (sample_leaf_removal) {
        add_leaf_removal_samples(slide, H, rng);
      }
    }
};
