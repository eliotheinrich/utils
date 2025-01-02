#pragma once

#include "EntropyState.hpp"

#include <Frame.h>

#include <cstdint>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <memory>

class EntropySampler {
  public:
    EntropySampler(dataframe::ExperimentParams &params) {  
      renyi_indices = parse_renyi_indices(dataframe::utils::get<std::string>(params, "renyi_indices", "2"));

      system_size = dataframe::utils::get<int>(params, "system_size");

      spacing = dataframe::utils::get<int>(params, "spacing", 1);

      sample_entropy = dataframe::utils::get<int>(params, "sample_entropy", false);
      partition_size = dataframe::utils::get<int>(params, "partition_size", system_size/2);

      sample_all_partition_sizes = dataframe::utils::get<int>(params, "sample_all_partition_sizes", false);
      spatially_average = dataframe::utils::get<int>(params, "spatial_avg", true);

      sample_mutual_information = dataframe::utils::get<int>(params, "sample_mutual_information", false);
      if (sample_mutual_information) {
        assert(partition_size > 0);
        num_eta_bins = dataframe::utils::get<int>(params, "num_mi_bins", 100);
        min_eta = dataframe::utils::get<double>(params, "min_eta", 0.01);
        max_eta = dataframe::utils::get<double>(params, "max_eta", 1.0);
      }

      sample_fixed_mutual_information = dataframe::utils::get<int>(params, "sample_fixed_mutual_information", false);
      if (sample_fixed_mutual_information) {
        x1 = dataframe::utils::get<int>(params, "x1");
        x2 = dataframe::utils::get<int>(params, "x2");
        x3 = dataframe::utils::get<int>(params, "x3");
        x4 = dataframe::utils::get<int>(params, "x4");
      }

      sample_variable_mutual_information = dataframe::utils::get<int>(params, "sample_variable_mutual_information", false);

      sample_correlation_distance = dataframe::utils::get<int>(params, "sample_correlation_distance", false);
      if (sample_correlation_distance) {
        num_distance_bins = dataframe::utils::get<int>(params, "num_distance_bins", 100);
      }

      // Spatial average is generally applied when boundary conditions are periodic, so spatially_average
      // as a default for setting pbc used by variable mutual information samples
      pbc = dataframe::utils::get<int>(params, "pbc", spatially_average);
    }

    ~EntropySampler()=default;

    void add_entropy_samples(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<double> entropy_samples;
      if (spatially_average) {
        entropy_samples = spatial_entropy_samples(partition_size, index, state);
      } else {
        entropy_samples = {state->cum_entropy(partition_size, index)};
      }
      std::string key = "entropy" + std::to_string(index) + "_" + std::to_string(partition_size);
      dataframe::utils::emplace(samples, key, entropy_samples);
    }

    void add_mutual_information_samples(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<double> entropy_table = compute_entropy_table(index, state);
      std::vector<std::vector<double>> mutual_information_samples(num_eta_bins);
      for (uint32_t x1 = 0; x1 < system_size; x1++) {
        for (uint32_t x3 = 0; x3 < system_size; x3++) {
          uint32_t x2 = (x1 + partition_size) % system_size;
          uint32_t x4 = (x3 + partition_size) % system_size;

          double eta = get_eta(x1, x2, x3, x4);
          if (!(eta > min_eta) || !(eta < max_eta)) {
            continue;
          }

          std::vector<uint32_t> combined_sites = to_combined_interval(x1, x2, x3, x4);

          double entropy1 = entropy_table[x1];
          double entropy2 = entropy_table[x3];
          double entropy3 = state->entropy(combined_sites, index);

          double sample = entropy1 + entropy2 - entropy3;
          uint32_t idx = get_bin_idx(eta, min_eta, max_eta, num_eta_bins);

          mutual_information_samples[idx].push_back(sample);
        }
      }
      
      std::string key = "mutual_information" + std::to_string(index);
      dataframe::utils::emplace(samples, key, mutual_information_samples);
    }

    void add_fixed_mutual_information_samples(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<uint32_t> interval1 = to_interval(x1, x2);
      std::vector<uint32_t> interval2 = to_interval(x3, x4);
      std::vector<uint32_t> interval3 = to_combined_interval(x1, x2, x3, x4);

      std::string key = "mutual_information" + std::to_string(index);
      auto mutual_information = state->entropy(interval1, index) + state->entropy(interval2, index) - state->entropy(interval3, index);
      dataframe::utils::emplace(samples, key, mutual_information);
    }

    void add_variable_mutual_information_samples(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<std::vector<double>> mutual_information_samples(system_size/2);
      for (uint32_t i = 0; i < system_size/2; i++) {
        std::vector<uint32_t> sites(2*i);
        if (pbc) {
          for (uint32_t j = 0; j < i; j++) {
            sites[2*j] = j;
            sites[2*j+1] = j + system_size/2;
          }
        } else {
          for (uint32_t j = 0; j < i; j++) {
            sites[2*j] = j;
            sites[2*j+1] = system_size - j - 1;
          }
        }

        // CHECK THIS
        if (spatially_average) {
          mutual_information_samples[i] = spatial_entropy_samples(sites, index, state);
        } else {
          mutual_information_samples[i] = {state->entropy(sites, index)};
        }
      }

      dataframe::utils::emplace(samples, "variable_mutual_information", mutual_information_samples);
    }

    void add_entropy_all_partition_sizes(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<std::vector<double>> entropy_samples(system_size);
      for (uint32_t i = 0; i < system_size; i++) {
        std::vector<double> s;
        if (spatially_average) {
          s = spatial_entropy_samples(i, index, state);
        } else {
          s = {state->cum_entropy(i, index)};
        }

        entropy_samples[i] = s;
      }

      std::string key = "entropy" + std::to_string(index);
      dataframe::utils::emplace(samples, key, entropy_samples);
    }

    void add_correlation_distance_samples(dataframe::SampleMap &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<std::vector<double>> bins(num_distance_bins, std::vector<double>());
      std::vector<size_t> counts(num_distance_bins, 0);
      for (size_t x1 = 0; x1 < system_size; x1++) {
        for (size_t x2 = 0; x2 < system_size; x2++) {
          if (x1 == x2) {
            continue;
          }

          std::vector<uint32_t> s1{static_cast<uint32_t>(x1)};
          std::vector<uint32_t> s2{static_cast<uint32_t>(x2)};
          std::vector<uint32_t> s3{static_cast<uint32_t>(x1), static_cast<uint32_t>(x2)};

          double I = state->entropy(s1, index) + state->entropy(s2, index) - state->entropy(s3, index);
          double d = distance(x1, x2);
          size_t i = get_bin_idx(d, 0.0, 1.0, num_distance_bins);
          bins[i].push_back(I);
        }
      }
      
      std::string key = "correlation_distance" + std::to_string(index);
      dataframe::utils::emplace(samples, key, bins);
    }

    void add_samples(dataframe::SampleMap &samples, std::shared_ptr<EntropyState> state) {
      for (auto const &i : renyi_indices) {
        if (sample_entropy) {
          add_entropy_samples(samples, i, state);
        }

        if (sample_all_partition_sizes) {
          add_entropy_all_partition_sizes(samples, i, state);
        }

        if (sample_mutual_information) {
          add_mutual_information_samples(samples, i, state);
        }

        if (sample_fixed_mutual_information) {
          add_fixed_mutual_information_samples(samples, i, state);
        }

        if (sample_variable_mutual_information) {
          add_variable_mutual_information_samples(samples, i, state);
        }

        if (sample_correlation_distance) {
          add_correlation_distance_samples(samples, i, state);
        }
      }
    }

  protected:
    uint32_t system_size;

    std::vector<uint32_t> renyi_indices;

    uint32_t partition_size;
    uint32_t spacing;
    bool spatially_average;
    bool sample_entropy;

    bool sample_all_partition_sizes;

    size_t num_eta_bins;
    double min_eta;
    double max_eta;
    bool sample_mutual_information;

    uint32_t x1;
    uint32_t x2;
    uint32_t x3;
    uint32_t x4;
    bool sample_fixed_mutual_information;

    bool pbc;
    bool sample_variable_mutual_information;
    bool sample_correlation_distance;
    size_t num_distance_bins;

    size_t get_bin_idx(double s, double min, double max, size_t num_bins) const {
      if ((s < min) || (s > max)) {
        std::string error_message = std::to_string(s) + " is not between " + std::to_string(min) + " and " + std::to_string(max) + ". \n";
        throw std::invalid_argument(error_message);
      }

      double bin_width = static_cast<double>(max - min)/num_bins;
      return static_cast<size_t>((s - min) / bin_width);
    }

    std::vector<uint32_t> to_interval(uint32_t x1, uint32_t x2) const {
      if (!(x1 < system_size && x2 <= system_size)) {
        throw std::invalid_argument("Invalid x1 or x2 passed to EntropyState.to_interval().");
      }

      if (x2 == system_size) x2 = 0;
      std::vector<uint32_t> interval;
      uint32_t i = x1;
      while (true) {
        interval.push_back(i);
        i = (i + 1) % system_size;
        if (i == x2) {
          return interval;
        }
      }
    }

    std::vector<uint32_t> to_combined_interval(int x1, int x2, int x3, int x4) const {
      std::vector<uint32_t> interval1 = to_interval(x1, x2);
      std::sort(interval1.begin(), interval1.end());
      std::vector<uint32_t> interval2 = to_interval(x3, x4);
      std::sort(interval2.begin(), interval2.end());
      std::vector<uint32_t> combined_interval;


      std::set_union(interval1.begin(), interval1.end(), 
          interval2.begin(), interval2.end(), 
          std::back_inserter(combined_interval));

      return combined_interval;
    }

    double get_x(int x1, int x2) const {
      return std::sin(std::abs(x1 - x2)*M_PI/system_size);
    }

    double get_eta(int x1, int x2, int x3, int x4) const {
      double x12 = get_x(x1, x2);
      double x34 = get_x(x3, x4);
      double x13 = get_x(x1, x3);
      double x24 = get_x(x2, x4);
      return x12*x34/(x13*x24);
    }

    double distance(int x1, int x2) const {
      double d = std::abs(x1 - x2)/static_cast<double>(system_size);
      if (pbc) {
        return (d > 0.5) ? (1.0 - d) : d;
      } else {
        return d;
      }
    }

    std::vector<double> compute_entropy_table(uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<double> table;

      for (uint32_t x1 = 0; x1 < system_size; x1++) {
        uint32_t x2 = (x1 + partition_size) % system_size;

        std::vector<uint32_t> sites = to_interval(x1, x2);
        table.push_back(state->entropy(sites, index));
      }

      return table;
    }

    std::vector<double> spatial_entropy_samples(const std::vector<uint32_t> &sites, uint32_t index, std::shared_ptr<EntropyState> state) const {
      uint32_t num_partitions = std::max((system_size - partition_size)/spacing, 1u);

      std::vector<double> samples(num_partitions);
      std::vector<uint32_t> offset_sites(sites.size());
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::transform(sites.begin(), sites.end(), offset_sites.begin(), 
          [i, this](uint32_t x) { return (x + i*spacing) % system_size; }
        );

        samples[i] = state->entropy(offset_sites, index);
      }

      return samples;
    }

    std::vector<double> spatial_entropy_samples(uint32_t partition_size, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<uint32_t> sites(partition_size);
      std::iota(sites.begin(), sites.end(), 0);

      return spatial_entropy_samples(sites, index, state);
    }

    std::vector<double> spatial_entropy_samples(uint32_t index, std::shared_ptr<EntropyState> state) const {
      return spatial_entropy_samples(partition_size, index, state);
    }

  private:
    // Expects a list of indices in the format "1,2,3"
    static std::vector<uint32_t> parse_renyi_indices(const std::string &renyi_indices_str) {
      std::vector<uint32_t> indices;
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
