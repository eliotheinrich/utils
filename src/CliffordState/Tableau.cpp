#include "Tableau.hpp"
#include "QuantumCHPState.hpp"

#include <glaze/glaze.hpp>

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    "tableau", &QuantumCHPState::tableau
  );
};

template<>
struct glz::meta<Tableau> {
  static constexpr auto value = glz::object(
    "num_qubits", &Tableau::num_qubits,
    "track_destabilizers", &Tableau::track_destabilizers,
    "rows", &Tableau::rows
  );
};

template<>
struct glz::meta<PauliString> {
  static constexpr auto value = glz::object(
    "num_qubits", &PauliString::num_qubits,
    "phase", &PauliString::phase,
    "width", &PauliString::width,
    "bit_string", &PauliString::bit_string
  );
};
