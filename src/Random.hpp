#pragma once

#include <random>

class Random {
  public:
    static Random& get_instance() {
      thread_local static Random instance;
      return instance;
    }
  private:
    uint32_t seed;
    std::minstd_rand rng;
    Random() {
      thread_local std::random_device gen;
      seed = gen();
      rng.seed(seed);
    } 

  public:
    Random(const Random&) = delete;
    Random& operator=(const Random&) = delete;

    static void seed_rng(uint32_t s) {
      Random& instance = get_instance();
      instance.seed = s;
      instance.rng.seed(s);
    }

    static uint32_t get_seed() {
      return Random::get_instance().seed;
    }

    uint32_t rand() {
      return rng();
    }
};

inline static uint32_t randi() {
  return Random::get_instance().rand();
}

inline static double randf() {
  return static_cast<double>(randi())/static_cast<double>(RAND_MAX);
}
