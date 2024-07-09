#include "SimulatorDisplay.h"
#include <unistd.h>

#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

class SimulatorDisplay::InputProcessImpl {
  private:
    using KeyMap = std::map<int, std::function<void(SimulatorDisplay&, GLFWwindow*)>>;
    KeyMap keymap;
    std::map<int, double> prevtime;

  public:
    InputProcessImpl() {
      keymap[GLFW_KEY_SPACE] = [](SimulatorDisplay& sim, GLFWwindow* window) { sim.paused = !sim.paused; };
      keymap[GLFW_KEY_ESCAPE] = [](SimulatorDisplay& sim, GLFWwindow* window) { glfwSetWindowShouldClose(window, true); };
    }

    double update_key_time(int key_code, double time) {
      if (!prevtime.contains(key_code)) {
        prevtime[key_code] = 0.0;
      }

      double t = time - prevtime[key_code];
      prevtime[key_code] = time;
      return t;
    }

    auto get_key_callback() {
      auto callback = [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto& self = *static_cast<SimulatorDisplay*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_RELEASE) {
          if (self.input_processor->keymap.contains(key)) {
            auto func = self.input_processor->keymap[key];
            func(self, window);
          }

          self.simulator->callback(key);
        }
      };

      return callback;
    }
};

SimulatorDisplay::SimulatorDisplay(std::unique_ptr<Drawable> state, size_t steps_per_update, size_t fps) : steps_per_update(steps_per_update), fps(fps) {
  // Setting up state variables
  simulator = std::move(state);
  paused = false;
  background_color = {0.0, 0.0, 0.0, 0.0};

  input_processor = std::make_unique<InputProcessImpl>();
}

void SimulatorDisplay::init_buffers() {
  // Setting up GL variables
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glGenTextures(1, &texture_idx); 

  // Configure vertex attributes
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  float vertices[] = {
    // positions         // texture coords
    1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
    1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
  };

  unsigned int indices[] = {
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
  };

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);

  // Configure textures
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_idx);  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Unbind buffers
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);  

  // Activate shader
  shader = Shader("/Users/eliotheinrich/Projects/utils/src/SimulatorDisplay/", "vertex_texture_shader.vs", "fragment_texture_shader.fs");
  shader.set_int("tex", 0);
}

SimulatorDisplay::~SimulatorDisplay() {
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
}


void SimulatorDisplay::animate(size_t width, size_t height) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow* window = glfwCreateWindow(width, height, "Simulator", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return;
  }
  glfwMakeContextCurrent(window); 

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return;
  }  

  init_buffers();

  glViewport(0, 0, width, height);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  

  glfwSetWindowUserPointer(window, this);
  auto key_callback = input_processor->get_key_callback();
  glfwSetKeyCallback(window, key_callback);

  double t1, t2, dt;
  double target_dt = 1.0/fps;

  std::vector<double> times(100);
  while (!glfwWindowShouldClose(window)) {
    t1 = glfwGetTime();

    glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Do physics and poll texture
    if (!paused) {
      simulator->timesteps(steps_per_update);
    }
    Texture texture = simulator->get_texture();

    // Draw texture
    shader.use();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindTexture(GL_TEXTURE_2D, texture_idx);  
    glActiveTexture(GL_TEXTURE0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.n, texture.m, 0, GL_RGBA, GL_FLOAT, texture.data());

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();    

    // Fix framerate
    t2 = glfwGetTime();
    dt = t2 - t1;

    times.push_back(dt);
    times.erase(times.begin());
    double avg = 0.0;
    for (auto t : times) {
      avg += t;
    }
    //std::cout << fmt::format("dt = {}\n", avg/100);
    if (dt < target_dt) {
      usleep((target_dt - dt)*1e6);
    }
  }

  glfwTerminate();
}
