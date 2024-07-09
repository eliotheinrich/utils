#pragma once

#include <vector>
#include <Frame.h>

#include <fmt/format.h>

struct Color {
  float r;
  float g;
  float b;
  float w;
};

template <>
struct std::formatter<Color> : std::formatter<std::string> {
  auto format(Color c, format_context& ctx) const {
    return formatter<string>::format(
      std::format("{}, {}, {}, {}", c.r, c.g, c.b, c.w), ctx);
  }
};

class Texture {
  private:

  public:
    size_t n;
    size_t m;
    std::vector<float> texture;

    Texture(size_t n, size_t m) : n(n), m(m), texture(4*n*m) {}
    Texture()=default;

    void set(size_t i, size_t j, Color color) {
      size_t idx = 4*(i + j*m);
      texture[idx]   = color.r;
      texture[idx+1] = color.g;
      texture[idx+2] = color.b;
      texture[idx+3] = color.w;
    }

    void* data() const {
      return (void*) texture.data();
    }
};

class Drawable : public dataframe::Simulator {
  public:
    using dataframe::Simulator::Simulator;

    virtual void callback(int key) {
      return;
    }

    virtual Texture get_texture() const {
      return Texture();
    }
};
