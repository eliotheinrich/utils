#pragma once

#include <random>
#include <iostream>

#include <dataframe/Frame.h>

struct Color {
  float r;
  float g;
  float b;
  float w;
};

template <>
struct fmt::formatter<Color> : fmt::formatter<std::string> {
  auto format(const Color& c, fmt::format_context& ctx) const {
    return fmt::formatter<std::string>::format(
      fmt::format("{}, {}, {}, {}", c.r, c.g, c.b, c.w), ctx);
  }
};

class Texture {
  public:
    size_t n;
    size_t m;
    std::vector<float> texture;

    Texture(const std::vector<float>& data, size_t n, size_t m) : n(n), m(m), texture(data) {
      if (len() != data.size()) {
        throw std::runtime_error("Invalid texture dimensions passed to Texture.");
      }
    }

    Texture(size_t n, size_t m) : n(n), m(m), texture(4*n*m) {}

    Texture(const std::vector<std::vector<std::vector<float>>>& data) {
      n = data.size();
      if (n > 0) {
        m = data[0].size();
        for (size_t k = 0; k < n; k++) {
          if (data[k].size() != m) {
            throw std::runtime_error("Data not square.");
          }
          for (size_t l = 0; l < m; l++) {
            if (data[k][l].size() != 4) {
              throw std::runtime_error("Data is not RGBA formatted.");
            }
          }
        }
      }

      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
          set(i, j, {data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3]});
        }
      }
    }

    Texture()=default;

    void set(size_t i, size_t j, Color color) {
      size_t idx = 4*(i + j*m);
      texture[idx]   = color.r;
      texture[idx+1] = color.g;
      texture[idx+2] = color.b;
      texture[idx+3] = color.w;
    }
    
    size_t len() const {
      return 4*n*m;
    }

    const float* data() const {
      return texture.data();
    }
};

class Simulator {
  public:
    Simulator()=default;

    Simulator(dataframe::ExperimentParams &params) {}

    virtual ~Simulator()=default;

    virtual void timesteps(uint32_t num_steps)=0;

    // By default, do nothing special during equilibration timesteps
    // May want to include, i.e., annealing 
    virtual void equilibration_timesteps(uint32_t num_steps) {
      timesteps(num_steps);
    }

    virtual dataframe::SampleMap take_samples() const {
      return dataframe::SampleMap();
    }

    virtual std::vector<dataframe::byte_t> serialize() const {
      std::cerr << "WARNING: serialize not implemented for this simulator; returning empty data.";
      return {};
    }

    virtual void deserialize(const std::vector<dataframe::byte_t>& data) {
      std::cerr << "Deserialize not implemented for this simulator; skipping.";
    }

    virtual void key_callback(int key) {
      return;
    }

    virtual void annealing_callback(int epoch, int num_epochs) {

    }

    virtual Texture get_texture() const {
      throw std::runtime_error("Called get_texture on a C++ simulator that does not implement it.");
    }

  protected:
    std::minstd_rand rng;
};
