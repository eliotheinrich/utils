#pragma once

#include <Frame.h>

#include <cstdint>

class InterfaceSampler {
  private:	
    uint32_t system_size;

    bool sample_surface;
    bool sample_surface_avg;

    bool sample_structure_function;
    bool transform_fluctuations;

    uint32_t max_width;
    bool sample_rugosity;
    bool sample_roughness;

    bool sample_staircases;

    bool sample_threshold;
    int threshold;

    bool sample_edges;

    uint32_t num_bins;
    uint32_t min_av;
    uint32_t max_av;
    bool sample_avalanche_sizes;
    std::vector<uint32_t> avalanche_sizes;

    uint32_t get_bin_idx(double s) const {
      if ((s < min_av) || (s > max_av)) {
        std::string error_message = std::to_string(s) + " is not between " + std::to_string(min_av) + " and " + std::to_string(max_av) + ". \n";
        throw std::invalid_argument(error_message);
      }

      double bin_width = static_cast<double>(max_av - min_av)/num_bins;
      uint32_t idx = static_cast<uint32_t>((s - min_av) / bin_width);
      return idx;
    }

  public:
    InterfaceSampler(dataframe::ExperimentParams& params) {
      system_size = dataframe::utils::get<int>(params, "system_size");

      sample_surface = dataframe::utils::get<int>(params, "sample_surface", false);
      sample_surface_avg = dataframe::utils::get<int>(params, "sample_surface_avg", false);

      max_width = dataframe::utils::get<int>(params, "max_width", system_size/2);
      sample_rugosity = dataframe::utils::get<int>(params, "sample_rugosity", false);
      sample_roughness = dataframe::utils::get<int>(params, "sample_roughness", false);

      sample_structure_function = dataframe::utils::get<int>(params, "sample_structure_function", false);
      transform_fluctuations = dataframe::utils::get<int>(params, "transform_fluctuations", false);

      sample_avalanche_sizes = dataframe::utils::get<int>(params, "sample_avalanche_sizes", false);

      num_bins = dataframe::utils::get<int>(params, "num_bins", 100);
      min_av = dataframe::utils::get<int>(params, "min_av", 1);
      max_av = dataframe::utils::get<int>(params, "max_av", 100);

      if (max_av <= min_av) {
        throw std::invalid_argument("max_av must be greater than min_av");
      }

      avalanche_sizes = std::vector<uint32_t>(num_bins);

      sample_staircases = dataframe::utils::get<int>(params, "sample_staircases", false);

      sample_threshold = dataframe::utils::get<int>(params, "sample_threshold", false);
      if (sample_threshold) {
        threshold = dataframe::utils::get<int>(params, "threshold", 0u);
      }

      sample_edges = dataframe::utils::get<int>(params, "sample_edges", false);
    }

    ~InterfaceSampler()=default;

    void record_size(uint32_t s) {
      if (s >= min_av && s < max_av) {
        uint32_t idx = get_bin_idx(s);
        avalanche_sizes[idx]++;
      }
    }

    double roughness(const std::vector<int> &surface) const {
      size_t num_sites = surface.size();

      return roughness_window(num_sites/2, surface);
    }

    double roughness_window(uint32_t width, const std::vector<int>& surface) const {
      double w = 0.0;
      double hb = surface_avg_window(width, surface);

      size_t num_sites = surface.size();
      for (size_t i = num_sites/2 - width; i < num_sites/2 + width; i++) {
        w += std::pow(surface[i] - hb, 2);
      }

      return w/(2.0*width);
    }

    double surface_avg(const std::vector<int>& surface) const {
      size_t num_sites = surface.size();

      return surface_avg_window(num_sites/2, surface);
    }

    double surface_avg_window(uint32_t width, const std::vector<int> &surface) const {
      size_t num_sites = surface.size();
      if (2*width > num_sites) {
        std::string error_message = "width = " + std::to_string(width) + 
          " is too large for the size of the surface = " + std::to_string(num_sites) + ".";
        throw std::invalid_argument(error_message);
      }

      double sum = 0.0;
      for (uint32_t i = num_sites/2 - width; i < num_sites/2 + width; i++) {
        sum += surface[i];
      }

      return sum/(2.0*width);
    }

    void add_surface_samples(dataframe::data_t &samples, const std::vector<int>& surface) const {
      size_t num_sites = surface.size();

      std::vector<double> surface_d(num_sites);
      std::transform(surface.begin(), surface.end(), surface_d.begin(),
                     [](int v) { return static_cast<double>(v); });
      dataframe::utils::emplace(samples, "surface", surface_d);
    }

    void add_avalanche_samples(dataframe::data_t &samples) {
      uint32_t total_avalanches = 0;
      for (uint32_t i = 0; i < num_bins; i++) {
        total_avalanches += avalanche_sizes[i];
      }

      std::vector<double> avalanche_prob(num_bins, 0.0);

      if (total_avalanches != 0) {
        for (uint32_t i = 0; i < num_bins; i++) {
          avalanche_prob[i] = static_cast<double>(avalanche_sizes[i])/total_avalanches;
        }
      }

      dataframe::utils::emplace(samples, "avalanche", avalanche_prob);

      avalanche_sizes = std::vector<uint32_t>(num_bins, 0);
    }


    std::vector<double> structure_function(const std::vector<int>& surface) const;
  
    void add_structure_function_samples(dataframe::data_t &samples, const std::vector<int> &surface) const {
      std::vector<double> sk = structure_function(surface);
      dataframe::utils::emplace(samples, "structure", sk);
    }

    void add_rugosity_samples(dataframe::data_t& samples, const std::vector<int>& surface) const {
      uint32_t num_sites = surface.size();
      size_t size = std::min(num_sites/2, max_width) - 1;

      std::vector<double> rugosity(size);
      for (uint32_t width = 1; width < size + 1; width++) {
        rugosity[width-1] = std::pow(roughness_window(width, surface), 2);
      }

      dataframe::utils::emplace(samples, "rugosity", rugosity);
    }

    inline bool staircase(const size_t i, const std::vector<int>& surface) const {
      int d1 = surface[i+1] - surface[i];
      int d2 = surface[i] - surface[i-1];

      return (d1 == d2) && (std::abs(d1) == 1);
    }

    void add_staircase_samples(dataframe::data_t& samples, const std::vector<int>& surface) const {
      size_t num_sites = surface.size();

      std::vector<double> staircase_counts(num_sites, 0.0);

      bool sizing_staircase = false;
      uint32_t size = 0;
      for (size_t i = 1; i < num_sites - 1; i++) {
        if (staircase(i, surface)) {
          sizing_staircase = true;
          size++;
        } else {
          if (sizing_staircase) {
            staircase_counts[size]++;
            size = 0;
          }

          sizing_staircase = false;
        } 
      }

      if (size) {
        staircase_counts[size]++;
      }

      dataframe::utils::emplace(samples, "staircases", staircase_counts);
    }

    void add_threshold_samples(dataframe::data_t& samples, const std::vector<int>& surface) const {
      double m = 0.0;
      size_t num_sites = surface.size();
      for (size_t i = 0; i < num_sites; i++) {
        if (surface[i] <= threshold) {
          m += 1.0;
        }
      }

      dataframe::utils::emplace(samples, "threshold_fraction", m/num_sites);
    }
    
    std::pair<size_t, size_t> find_edges(const std::vector<int>& surface) const {
      size_t num_sites = surface.size();

      std::vector<double> staircase_counts(num_sites, 0.0);

      size_t i = 1;
      while (i < num_sites && staircase(i, surface)) {
        i++;
      }

      double e1 = i;

      i = num_sites - 1;

      while (i > 0 && staircase(i, surface)) {
        i--;
      }

      double e2 = i;

      return {e1, e2};
    }

    void add_edge_samples(dataframe::data_t& samples, const std::vector<int>& surface) const {
      auto [e1, e2] = find_edges(surface);
      dataframe::utils::emplace(samples, "edge_positions", {static_cast<double>(e1), static_cast<double>(e2)});
    }


    void add_samples(dataframe::data_t &samples, const std::vector<int>& surface) {
      if (sample_surface) {
        add_surface_samples(samples, surface);
      }

      if (sample_surface_avg) {
        dataframe::utils::emplace(samples, "surface_avg", surface_avg(surface));
      }

      if (sample_rugosity) {
        add_rugosity_samples(samples, surface);
      }

      if (sample_roughness) {
        dataframe::utils::emplace(samples, "roughness", roughness(surface));
      }

      if (sample_avalanche_sizes) {
        add_avalanche_samples(samples);
      }

      if (sample_structure_function) {
        add_structure_function_samples(samples, surface);
      }

      if (sample_staircases) {
        add_staircase_samples(samples, surface);
      }

      if (sample_threshold) {
        add_threshold_samples(samples, surface);
      }

      if (sample_edges) {
        add_edge_samples(samples, surface);
      }
    }
};
