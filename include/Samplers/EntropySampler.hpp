#pragma once

#include <Frame.h>

#include <cstdint>
#include <sstream>
#include <algorithm>
#include <numeric>

class EntropyState {
  protected:
    uint32_t system_size;

  public:
    EntropyState()=default;

    EntropyState(uint32_t system_size) : system_size(system_size) {}

    virtual double entropy(const std::vector<uint32_t>& sites, uint32_t index)=0;

    template <typename T = double>
    T cum_entropy(uint32_t i, uint32_t index = 2u, bool direction = true) {
      if (direction) { // Left-oriented cumulative entropy
        std::vector<uint32_t> sites(i+1);
        std::iota(sites.begin(), sites.end(), 0);
        return static_cast<T>(entropy(sites, index));
      } else { // Right-oriented cumulative entropy
        std::vector<uint32_t> sites(system_size - i);
        std::iota(sites.begin(), sites.end(), i);
        return static_cast<T>(entropy(sites, index));
      }
    }

    template <typename T = double>
    std::vector<T> get_entropy_surface(uint32_t index=2u) {
      std::vector<T> entropy_surface(system_size);

      for (uint32_t i = 0; i < system_size; i++) {
        entropy_surface[i] = cum_entropy<T>(i, index);
      }

      return entropy_surface;
    }
};

class EntropySampler {
  public:
    EntropySampler(dataframe::Params &params) {  
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
        num_bins = dataframe::utils::get<int>(params, "num_mi_bins", 100);
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
      // Spatial average is generally applied when boundary conditions are periodic, so spatially_average
      // as a default for setting pbc used by variable mutual information samples
      pbc = dataframe::utils::get<int>(params, "pbc", spatially_average);
    }

    void add_entropy_samples(dataframe::data_t &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      dataframe::Sample s = spatially_average ? spatially_averaged_entropy(partition_size, index, state) : state->cum_entropy(partition_size, index);
      samples.emplace("entropy" + std::to_string(index) + "_" + std::to_string(partition_size), s);
    }

    void add_mutual_information_samples(dataframe::data_t &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<double> entropy_table = compute_entropy_table(index, state);
      std::vector<dataframe::Sample> mutual_information_samples(num_bins, dataframe::Sample());
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
          uint32_t idx = get_bin_idx(eta);

          mutual_information_samples[idx] = mutual_information_samples[idx].combine(sample); 
        }
      }

      samples.emplace("mutual_information" + std::to_string(index), mutual_information_samples);
    }

    void add_fixed_mutual_information_samples(dataframe::data_t &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<uint32_t> interval1 = to_interval(x1, x2);
      std::vector<uint32_t> interval2 = to_interval(x3, x4);
      std::vector<uint32_t> interval3 = to_combined_interval(x1, x2, x3, x4);

      samples.emplace("mutual_information" + std::to_string(index), 
          state->entropy(interval1, index) + state->entropy(interval2, index) - state->entropy(interval3, index));

      //uint32_t num_partitions = std::max(system_size/spacing, 1u);

      //dataframe::Sample sample;
      //std::vector<uint32_t> offset_sites(sites.size());
      //for (uint32_t i = 0; i < num_partitions; i++) {
      //  std::transform(sites.begin(), sites.end(), offset_sites.begin(), 
      //    [i, this](uint32_t x) { return (x + i*spacing) % system_size; }
      //  );

      //  sample = sample.combine(state->entropy(offset_sites, index));
      //}

      //return sample;
    }

    void add_variable_mutual_information_samples(dataframe::data_t &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<dataframe::Sample> mutual_information_samples(system_size/2);
      for (uint32_t i = 0; i < system_size/2; i++) {
        dataframe::Sample sample;
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

        sample = spatially_average ? spatially_averaged_entropy(sites, index, state) : state->entropy(sites, index);

        mutual_information_samples[i] = sample;
      }

      samples.emplace("variable_mutual_information", mutual_information_samples);
    }

    void add_entropy_all_partition_sizes(dataframe::data_t &samples, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<dataframe::Sample> entropy_samples(system_size);
      for (uint32_t i = 0; i < system_size; i++) {
        dataframe::Sample s = spatially_average ? spatially_averaged_entropy(i, index, state) : state->cum_entropy(i, index);
        entropy_samples[i] = s;
      }

      samples.emplace("entropy" + std::to_string(index), entropy_samples);
    }

    void add_samples(dataframe::data_t &samples, std::shared_ptr<EntropyState> state) {
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

    uint32_t num_bins;
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

    uint32_t get_bin_idx(double s) const {
      if ((s < min_eta) || (s > max_eta)) {
        std::string error_message = std::to_string(s) + " is not between " + std::to_string(min_eta) + " and " + std::to_string(max_eta) + ". \n";
        throw std::invalid_argument(error_message);
      }

      double bin_width = static_cast<double>(max_eta - min_eta)/num_bins;
      return static_cast<uint32_t>((s - min_eta) / bin_width);
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

    std::vector<double> compute_entropy_table(uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<double> table;

      for (uint32_t x1 = 0; x1 < system_size; x1++) {
        uint32_t x2 = (x1 + partition_size) % system_size;

        std::vector<uint32_t> sites = to_interval(x1, x2);
        table.push_back(state->entropy(sites, index));
      }

      return table;
    }

    dataframe::Sample spatially_averaged_entropy(const std::vector<uint32_t> &sites, uint32_t index, std::shared_ptr<EntropyState> state) const {
      uint32_t num_partitions = std::max((system_size - partition_size)/spacing, 1u);

      dataframe::Sample sample;
      std::vector<uint32_t> offset_sites(sites.size());
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::transform(sites.begin(), sites.end(), offset_sites.begin(), 
          [i, this](uint32_t x) { return (x + i*spacing) % system_size; }
        );

        sample = sample.combine(state->entropy(offset_sites, index));
      }

      return sample;
    }

    dataframe::Sample spatially_averaged_entropy(uint32_t partition_size, uint32_t index, std::shared_ptr<EntropyState> state) const {
      std::vector<uint32_t> sites(partition_size);
      std::iota(sites.begin(), sites.end(), 0);

      uint32_t num_partitions = std::max((system_size - partition_size)/spacing, 1u);

      dataframe::Sample sample;
      std::vector<uint32_t> offset_sites(partition_size);
      for (uint32_t i = 0; i < num_partitions; i++) {
        std::transform(sites.begin(), sites.end(), offset_sites.begin(), 
          [i, this](uint32_t x) { return (x + i*spacing) % system_size; }
        );

        sample = sample.combine(state->entropy(offset_sites, index));
      }

      return sample;
    }

    dataframe::Sample spatially_averaged_entropy(uint32_t index, std::shared_ptr<EntropyState> state) const {
      return spatially_averaged_entropy(partition_size, index, state);
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

// TODO find a better place for this to live
static inline uint32_t mod(int a, int b) {
  int c = a % b;
  return (c < 0) ? c + b : c;
}
