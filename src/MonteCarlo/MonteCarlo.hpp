#pragma once

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <Frame.h>

#include <Drawable.h>

#define PI 3.14159265

enum BoundaryCondition { Periodic, Open };

typedef std::pair<uint32_t, int> Bond;

inline BoundaryCondition parse_boundary_condition(std::string s) {
  if (s == "periodic") {
    return BoundaryCondition::Periodic;
  } else if (s == "open") {
    return BoundaryCondition::Open;
  } else {
    std::string error_message = "Invalid boundary condition: " + s + "\n";
    throw std::invalid_argument(error_message);
  }
}

enum CoolingSchedule {
  Constant,
  Trig,
  Linear,
};

inline double const_T(int n, int n_max, double Ti, double Tf) {
  return 0.5*(Tf + Ti);
}

inline double trig_T(int n, int n_max, double Ti, double Tf) {
  return Ti + 0.5*(Tf - Ti)*(1 - cos(n*PI/n_max));
}

inline double linear_T(int n, int n_max, double Ti, double Tf) {
  return Ti - (Tf - Ti)*(n_max - n)/double(n_max);
}

static CoolingSchedule parse_cooling_schedule(const std::string& s) {
  if ((s == "constant") || (s == "const")) { 
    return CoolingSchedule::Constant; 
  } else if ((s == "trig") || (s == "cosine")) { 
    return CoolingSchedule::Trig; 
  } else if ((s == "linear")) { 
    return CoolingSchedule::Linear; 
  } else { 
    return CoolingSchedule::Constant; 
  }
}

class MonteCarloSimulator : public Drawable {
  // Most basic Monte-Carlo model to be simulated must have some notion of energy
  // as well as a mutation data structure. Specifics must be supplied by child classes.
  public:
    MonteCarloSimulator(dataframe::Params &params, uint32_t num_threads) : Drawable(params), num_threads(num_threads) {
      final_temperature = dataframe::utils::get<double>(params, "temperature");
      temperature = final_temperature;
      init_temperature = dataframe::utils::get<double>(params, "initial_temperature", final_temperature);
      num_cooling_updates = dataframe::utils::get<int>(params, "num_cooling_updates", 100);
      cooling_schedule = parse_cooling_schedule(dataframe::utils::get<std::string>(params, "cooling_schedule", "constant"));
    }
    virtual ~MonteCarloSimulator()=default;

    // Implement Simulator methods but introduce MonteCarlo methods
    virtual void timesteps(uint32_t num_steps) override {
      uint64_t num_updates = system_size()*num_steps;
      for (uint64_t i = 0; i < num_updates; i++) {
        generate_mutation();
        double dE = energy_change();

        double rf = randf();
        if (rf < std::exp(-dE/temperature)) {
          accept_mutation();
        } else {
          reject_mutation();
        }
      }
    }

    virtual void equilibration_timesteps(uint32_t num_steps) override {
      uint32_t steps_per_update = num_steps / num_cooling_updates;
      temperature = init_temperature;
      for (uint64_t i = 0; i < num_cooling_updates; i++) {
        timesteps(steps_per_update);
        switch (cooling_schedule) {
          case(CoolingSchedule::Constant) : temperature = const_T(i, num_cooling_updates, init_temperature, final_temperature);
          case(CoolingSchedule::Linear) : temperature = linear_T(i, num_cooling_updates, init_temperature, final_temperature);
          case(CoolingSchedule::Trig) : temperature = trig_T(i, num_cooling_updates, init_temperature, final_temperature);
        }
      }

      temperature = final_temperature;
      timesteps(num_steps % num_cooling_updates);
    }


    virtual void callback(int key) override;

    // To be overridden by child classes
    virtual double energy() const = 0;
    virtual double energy_change() = 0;
    virtual void generate_mutation() = 0;
    virtual void accept_mutation() = 0;
    virtual void reject_mutation() = 0;
    virtual uint64_t system_size() const = 0;

  protected:
    double temperature;

  private:
    uint32_t num_threads;

    CoolingSchedule cooling_schedule;
    uint32_t num_cooling_updates;
    double init_temperature;
    double final_temperature;
};

