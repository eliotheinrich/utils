#pragma once

#include <unistd.h>
#include <concepts>

#include <Frame.h>

#include <fmt/format.h>

#include "Drawable.h"
#include "Shader.h"

class SimulatorDisplay {
  public:
    class InputProcessImpl;
    std::unique_ptr<InputProcessImpl> input_processor;

    std::unique_ptr<Drawable> simulator;

    SimulatorDisplay(std::unique_ptr<Drawable> state, size_t steps_per_update=1, size_t fps=60);

    ~SimulatorDisplay();

    void animate(size_t width=900, size_t height=900);

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

    bool paused;
};
