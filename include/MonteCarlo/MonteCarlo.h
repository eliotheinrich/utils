#pragma once

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <Frame.h>

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

class MonteCarloSimulator : public dataframe::Simulator {
  // Most basic Monte-Carlo model to be simulated must have some notion of energy
  // as well as a mutation data structure. Specifics must be supplied by child classes.
  public:
    MonteCarloSimulator(dataframe::Params &params, uint32_t num_threads);
    virtual ~MonteCarloSimulator()=default;

    // Implement Simulator methods but introduce MCModel methods
    virtual void timesteps(uint32_t num_steps);
    virtual void equilibration_timesteps(uint32_t num_steps);

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

