#include "QuantumCircuit.h"
#include "PauliString.hpp"
#include <assert.h>

std::string QuantumCircuit::to_string() const {
	std::string s = "";
	for (auto const &inst : instructions) {
		s += std::visit(quantumcircuit_utils::overloaded{
			[](std::shared_ptr<Gate> gate) -> std::string {
				std::string gate_str = gate->label() + " ";
				for (auto const &q : gate->qbits) {
					gate_str += fmt::format("{} ", q);
				}

				return gate_str;
			},
			[](Measurement m) -> std::string {
				std::string meas_str = "mzr ";
				for (auto const &q : m.qbits) {
					meas_str += fmt::format("{} ", q);
				}

				return meas_str;
			}
		}, inst) + "\n";
	}

	return s;
}

uint32_t QuantumCircuit::num_params() const {
	uint32_t n = 0;
	for (auto const &inst : instructions) {
		n += std::visit(quantumcircuit_utils::overloaded{
			[](std::shared_ptr<Gate> gate) -> uint32_t { return gate->num_params(); },
			[](Measurement m) -> uint32_t { return 0u; }
		}, inst);
	}
	
	return n;
}

bool QuantumCircuit::is_clifford() const {
  for (auto const& inst : instructions) {
    bool valid = std::visit(quantumcircuit_utils::overloaded{
			[](std::shared_ptr<Gate> gate) -> uint32_t { return gate->is_clifford(); },
			[](Measurement m) -> uint32_t { return true; }
		}, inst);
    
    if (!valid) {
      return false;
    }
  }

  return true;
}

uint32_t QuantumCircuit::length() const {
  return instructions.size();
}

bool QuantumCircuit::contains_measurement() const {
  for (auto const &inst : instructions) {
    if (inst.index() == 1) {
      return true;
    }
  }

  return false;
}

void QuantumCircuit::apply_qubit_map(const std::vector<uint32_t>& qubits) {
  for (auto inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded{
			[&qubits](std::shared_ptr<Gate> gate) {
        std::vector<uint32_t> _qbits(gate->num_qubits);
        for (size_t q = 0; q < gate->num_qubits; q++) {
          _qbits[q] = qubits[gate->qbits[q]];
        }

        gate->qbits = _qbits;
			},
			[&qubits](Measurement m) { 
        std::vector<uint32_t> _qbits(m.qbits.size());
        for (size_t q = 0; q < m.qbits.size(); q++) {
          _qbits[q] = qubits[m.qbits[q]];
        }

        m.qbits = _qbits;
      }
		}, inst);
  }
}


void QuantumCircuit::add_instruction(const Instruction& inst) {
	instructions.push_back(inst);
}

void QuantumCircuit::add_measurement(uint32_t qubit) {
	std::vector<uint32_t> qargs{qubit};
	add_measurement(qargs);
}

void QuantumCircuit::add_measurement(const std::vector<uint32_t>& qargs) {
	add_measurement(Measurement(qargs));
}

void QuantumCircuit::add_measurement(const Measurement& m) {
	add_instruction(m);
}

void QuantumCircuit::add_gate(const std::shared_ptr<Gate> &gate) {
	add_instruction(gate);
}

void QuantumCircuit::add_gate(const std::string& name, const std::vector<uint32_t>& qbits) {
  add_gate(std::shared_ptr<Gate>(new SymbolicGate(name, qbits)));
}

void QuantumCircuit::add_gate(const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qbits) {
	if (!(gate.rows() == (1u << qbits.size()) && gate.cols() == (1u << qbits.size()))) {
		throw std::invalid_argument("Provided matrix does not have proper dimensions for number of qubits in circuit.");
	}

	add_gate(std::make_shared<MatrixGate>(gate, qbits));
}

void QuantumCircuit::add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	std::vector<uint32_t> qbits{qubit};
	add_gate(gate, qbits);
}

void QuantumCircuit::append(const QuantumCircuit& other) {
  if (num_qubits != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }
	for (auto const &inst : other.instructions) {
		add_instruction(inst);
	}
}

void QuantumCircuit::append(const QuantumCircuit& other, const std::vector<uint32_t>& qbits) {
  if (qbits.size() != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }

  QuantumCircuit qc_extended(other);
  qc_extended.resize(num_qubits);
  qc_extended.apply_qubit_map(qbits);

  append(qc_extended);
}

void QuantumCircuit::append(const Instruction& inst) {
  add_instruction(inst);
}

QuantumCircuit QuantumCircuit::bind_params(const std::vector<double>& params) const {
	if (params.size() != num_params()) {
		throw std::invalid_argument("Invalid number of parameters passed to bind_params.");
	}

	QuantumCircuit qc(num_qubits);

	uint32_t n = 0;
	for (auto const &inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded{
			[&qc, &n, &params](std::shared_ptr<Gate> gate) {
				std::vector<double> gate_params(gate->num_params());

				for (uint32_t i = 0; i < gate->num_params(); i++) {
					gate_params[i] = params[i + n];
				}
			
				n += gate->num_params();
				qc.add_gate(gate->define(gate_params), gate->qbits);
			},
			[&qc](Measurement m) { qc.add_measurement(m); }
		}, inst);
	}

	return qc;
}

QuantumCircuit QuantumCircuit::adjoint(const std::optional<std::vector<double>>& params_opt) const {
	bool params_passed = params_opt.has_value() && params_opt.value().size() != 0;

	if (params_passed) { // Params passed; check that they are valid and then perform adjoint.
		auto params = params_opt.value();
		if (params.size() != num_params()) {
			throw std::invalid_argument("Unbound parameters; adjoint cannot be defined.");
		}

		QuantumCircuit qc = bind_params(params);
		return qc.adjoint();
	} else if (!params_passed && num_params() == 0) { // No parameters to bind; go ahead and build adjoint
		QuantumCircuit qc(num_qubits);

		for (uint32_t i = 0; i < instructions.size(); i++) {
			std::visit(quantumcircuit_utils::overloaded{
				[&qc](std::shared_ptr<Gate> gate) { qc.add_gate(gate->adjoint()); },
				[&qc](Measurement m) { qc.add_measurement(m); }
			}, instructions[instructions.size() - i - 1]);
		}

		return qc;
	} else {
		throw std::invalid_argument("Params passed but nothing to bind.");
	}
}

Eigen::MatrixXcd QuantumCircuit::to_matrix(const std::optional<std::vector<double>>& params_opt) const {
	bool params_passed = params_opt.has_value() && params_opt.value().size() != 0;

	if (params_passed) { // Params passed; check that they are valid and then perform adjoint.
		auto params = params_opt.value();
		if (params.size() != num_params()) {
			throw std::invalid_argument("Unbound parameters; cannot convert circuit to matrix.");
		}

		QuantumCircuit qc = bind_params(params);
		return qc.to_matrix();
	} else if (!params_passed && num_params() == 0) { // No parameters to bind; go ahead and build adjoint

		Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
		Q.setIdentity();

		uint32_t p = num_qubits;

		for (uint32_t i = 0; i < instructions.size(); i++) {
			std::visit(quantumcircuit_utils::overloaded{
				[&Q, p](std::shared_ptr<Gate> gate) { Q = full_circuit_unitary(gate->define(), gate->qbits, p) * Q; },
				[](Measurement m) { throw std::invalid_argument("Cannot convert measurement to matrix."); }
			}, instructions[i]);
		}

		return Q;
	} else {
		throw std::invalid_argument("Params passed but nothing to bind.");
	}
}

// --- Library for building common circuits --- //
QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc, std::optional<int> seed) {
	thread_local std::random_device rd;
	std::mt19937 rng;
	if (seed.has_value()) {
		rng.seed(seed.value());
	} else {
		rng.seed(rd());
	}

	QuantumCircuit circuit(num_qubits);

	for (uint32_t i = 0; i < depth; i++) {
		for (uint32_t q = 0; q < num_qubits/2; q++) {
			auto [q1, q2] = get_targets(i, q, num_qubits);
			if (!pbc) {
				if (std::abs(int(q1) - int(q2)) > 1) {
					continue;
				}
			}

			circuit.add_gate(haar_unitary(2, rng), {q1, q2});
		}
	}

	return circuit;
}

QuantumCircuit hardware_efficient_ansatz(
		uint32_t num_qubits, 
		uint32_t depth, 
		const std::vector<std::string>& rotation_gates,
		const std::string& entangling_gate,
		bool final_layer
	) {

	QuantumCircuit circuit(num_qubits);

	for (uint32_t i = 0; i < depth; i++) {
		for (uint32_t q = 0; q < num_qubits/2; q++) {
			auto [q1, q2] = get_targets(i, q, num_qubits);

			for (auto const &s : rotation_gates) {
				auto gate = parse_gate(s, std::vector<uint32_t>{q1});
				assert(gate->num_qubits == 1);
				circuit.add_gate(gate);
				gate = parse_gate(s, std::vector<uint32_t>{q2});
				circuit.add_gate(gate);
			}

			auto entangler = parse_gate(entangling_gate, std::vector<uint32_t>{q1, q2});
			assert(entangler->num_qubits == 2);
			circuit.add_gate(entangler);
		}
	}

	if (final_layer) {
		for (uint32_t q = 0; q < num_qubits; q++) {
			for (auto const &s : rotation_gates) {
				auto gate = parse_gate(s, std::vector<uint32_t>{q});
				assert(gate->num_qubits == 1);
				circuit.add_gate(gate);
			}
		}
	}

	return circuit;
}

QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<std::vector<uint32_t>>& qargs_opt) {
	auto qargs = parse_qargs_opt(qargs_opt, num_qubits);

	QuantumCircuit circuit(num_qubits);
	
	for (auto const& q : qargs) {
		circuit.add_gate(std::make_shared<RxRotationGate>(std::vector<uint32_t>{q}));
	}
	
	return circuit;
}

// Returns the circuit which maps a PauliString to Z1 if z, otherwise to X1
QuantumCircuit single_qubit_random_clifford(uint32_t r) {
  QuantumCircuit qc(1);
  // r == 0 is identity, so do nothing in this case
  if (r == 1) {
    qc.add_gate("x", {0});
  } else if (r == 2) {
    qc.add_gate("y", {0});
  } else if (r == 3) {
    qc.add_gate("z", {0});
  } else if (r == 4) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
  } else if (r == 5) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("x", {0});
  } else if (r == 6) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("y", {0});
  } else if (r == 7) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("z", {0});
  } else if (r == 8) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
  } else if (r == 9) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("x", {0});
  } else if (r == 10) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("y", {0});
  } else if (r == 11) {
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("z", {0});
  } else if (r == 12) {
    qc.add_gate("h", {0});
  } else if (r == 13) {
    qc.add_gate("h", {0});
    qc.add_gate("x", {0});
  } else if (r == 14) {
    qc.add_gate("h", {0});
    qc.add_gate("y", {0});
  } else if (r == 15) {
    qc.add_gate("h", {0});
    qc.add_gate("z", {0});
  } else if (r == 16) {
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
  } else if (r == 17) {
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("x", {0});
  } else if (r == 18) {
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("y", {0});
  } else if (r == 19) {
    qc.add_gate("s", {0});
    qc.add_gate("h", {0});
    qc.add_gate("s", {0});
    qc.add_gate("z", {0});
  } else if (r == 20) {
    qc.add_gate("s", {0});
  } else if (r == 21) {
    qc.add_gate("s", {0});
    qc.add_gate("x", {0});
  } else if (r == 22) {
    qc.add_gate("s", {0});
    qc.add_gate("y", {0});
  } else if (r == 23) {
    qc.add_gate("s", {0});
    qc.add_gate("z", {0});
  }

  return qc;
}

// Performs an iteration of the random clifford algorithm outlined in https://arxiv.org/pdf/2008.06011.pdf
QuantumCircuit random_clifford_iteration(uint32_t num_qubits, std::minstd_rand& rng) {
  QuantumCircuit qc(num_qubits);

  // If only acting on one qubit, can easily lookup from a table
  if (num_qubits == 1) {
    return single_qubit_random_clifford(rng() % 24);
  }

  PauliString p1 = PauliString::rand(num_qubits, rng);
  PauliString p2 = PauliString::rand(num_qubits, rng);
  while (p1.commutes(p2)) {
    p2 = PauliString::rand(num_qubits, rng);
  }

  QuantumCircuit qc1 = p1.reduce(false);

  p2.evolve(qc1);

  PauliString z1p = PauliString::basis(num_qubits, "Z", 0, false);
  PauliString z1m = PauliString::basis(num_qubits, "Z", 0, true);

  if (p2 != z1p && p2 != z1m) {
    QuantumCircuit qc2 = p2.reduce(true);

    qc1.append(qc2);
  }

  return qc1;
}

QuantumCircuit random_clifford(uint32_t num_qubits, std::minstd_rand& rng) {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  QuantumCircuit qc(num_qubits);

  for (uint32_t i = 0; i < num_qubits; i++) {
    QuantumCircuit qc_iter = random_clifford_iteration(qubits.size(), rng);
    qc.append(qc_iter, qubits);
    qubits.pop_back();
  }

  return qc;
}

