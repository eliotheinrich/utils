#include "MonteCarlo.h"

#define DEFAULT_COOLING_SCHEDULE "constant"
#define DEFAULT_NUM_COOLING_UPDATES 100

MonteCarloSimulator::MonteCarloSimulator(dataframe::Params &params, uint32_t num_threads) : Simulator(params), num_threads(num_threads) {
  final_temperature = dataframe::utils::get<double>(params, "temperature");
  temperature = final_temperature;
  init_temperature = dataframe::utils::get<double>(params, "initial_temperature", final_temperature);
  num_cooling_updates = dataframe::utils::get<int>(params, "num_cooling_updates", DEFAULT_NUM_COOLING_UPDATES);
  cooling_schedule = parse_cooling_schedule(dataframe::utils::get<std::string>(params, "cooling_schedule", DEFAULT_COOLING_SCHEDULE));
}

void MonteCarloSimulator::timesteps(uint32_t num_steps) {
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

void MonteCarloSimulator::equilibration_timesteps(uint32_t num_steps) {
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
