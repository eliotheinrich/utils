#pragma once

#include <unistd.h>
#include <concepts>

#include <fmt/format.h>

#include <Simulator.hpp>

#include "Shader.h"

struct FrameData {
  int status_code;
  std::vector<int> keys;
};

class Animator {
  public:
    class InputProcessImpl;
    std::unique_ptr<InputProcessImpl> input_processor;
    
    class WindowImpl;
    std::unique_ptr<WindowImpl> window;

    Animator(const Color& background_color = {0.0, 0.0, 0.0, 0.0});
    ~Animator();

    void start(size_t width=900, size_t height=900);
    FrameData new_frame(const Texture& new_data);
    FrameData new_frame(const std::vector<float>& new_data, size_t n, size_t m);

    bool is_paused() const {
      return paused;
    }

  private:
    void init_buffers();

    size_t steps_per_update;
    size_t fps;
    bool serialize;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    unsigned int texture_idx;

    Shader shader;

    Color background_color;
    Texture frame_data;

    bool paused;
};
