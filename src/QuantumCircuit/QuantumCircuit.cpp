#include "QuantumCircuit.h"
#include "PauliString.hpp"
#include <assert.h>

#include <fmt/format.h>

std::string QuantumCircuit::to_string() const {
	std::string s = "";
	for (auto const &inst : instructions) {
		s += std::visit(quantumcircuit_utils::overloaded {
			[](std::shared_ptr<Gate> gate) -> std::string {
				std::string gate_str = gate->label() + " ";
				for (auto const &q : gate->qubits) {
					gate_str += fmt::format("{} ", q);
				}

				return gate_str;
			},
			[](Measurement m) -> std::string {
				std::string meas_str = "mzr ";
				for (auto const &q : m.qubits) {
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
		n += std::visit(quantumcircuit_utils::overloaded {
			[](std::shared_ptr<Gate> gate) -> uint32_t { return gate->num_params(); },
			[](Measurement m) -> uint32_t { return 0u; }
		}, inst);
	}
	
	return n;
}

bool QuantumCircuit::is_clifford() const {
  for (auto const& inst : instructions) {
    bool valid = std::visit(quantumcircuit_utils::overloaded {
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

void QuantumCircuit::apply_qubit_map(const Qubits& qubits) {
  for (auto inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded {
			[&qubits](std::shared_ptr<Gate> gate) {
        Qubits _qubits(gate->num_qubits);
        for (size_t q = 0; q < gate->num_qubits; q++) {
          _qubits[q] = qubits[gate->qubits[q]];
        }

        gate->qubits = _qubits;
			},
			[&qubits](Measurement m) { 
        Qubits _qubits(m.qubits.size());
        for (size_t q = 0; q < m.qubits.size(); q++) {
          _qubits[q] = qubits[m.qubits[q]];
        }

        m.qubits = _qubits;
      }
		}, inst);
  }
}


void QuantumCircuit::add_instruction(const Instruction& inst) {
	instructions.push_back(inst);
}

void QuantumCircuit::add_measurement(uint32_t qubit) {
	Qubits qubits{qubit};
	add_measurement(qubits);
}

void QuantumCircuit::add_measurement(const Qubits& qubits) {
	add_measurement(Measurement(qubits));
}

void QuantumCircuit::add_measurement(const Measurement& m) {
	add_instruction(m);
}

void QuantumCircuit::add_gate(const std::shared_ptr<Gate> &gate) {
	add_instruction(gate);
}

void QuantumCircuit::add_gate(const std::string& name, const Qubits& qubits) {
  add_gate(std::shared_ptr<Gate>(new SymbolicGate(name, qubits)));
}

void QuantumCircuit::add_gate(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
	if (!(gate.rows() == (1u << qubits.size()) && gate.cols() == (1u << qubits.size()))) {
		throw std::invalid_argument("Provided matrix does not have proper dimensions for number of qubits in circuit.");
	}

	add_gate(std::make_shared<MatrixGate>(gate, qubits));
}

void QuantumCircuit::add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	Qubits qubits{qubit};
	add_gate(gate, qubits);
}

void QuantumCircuit::append(const QuantumCircuit& other) {
  if (num_qubits != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }
	for (auto const &inst : other.instructions) {
		add_instruction(copy_instruction(inst));
	}
}

void QuantumCircuit::append(const QuantumCircuit& other, const Qubits& qubits) {
  if (qubits.size() != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }

  QuantumCircuit qc_extended(other);
  qc_extended.resize(num_qubits);
  qc_extended.apply_qubit_map(qubits);

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
		std::visit(quantumcircuit_utils::overloaded {
			[&qc, &n, &params](std::shared_ptr<Gate> gate) {
				std::vector<double> gate_params(gate->num_params());

				for (uint32_t i = 0; i < gate->num_params(); i++) {
					gate_params[i] = params[i + n];
				}
			
				n += gate->num_params();
				qc.add_gate(gate->define(gate_params), gate->qubits);
			},
			[&qc](Measurement m) { qc.add_measurement(m); }
		}, inst);
	}

	return qc;
}

void QuantumCircuit::random_clifford(const Qubits& qubits, std::minstd_rand& rng) {
  random_clifford_impl(qubits, rng, *this);
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
			std::visit(quantumcircuit_utils::overloaded {
				[&qc](std::shared_ptr<Gate> gate) { qc.add_gate(gate->adjoint()); },
				[&qc](Measurement m) { qc.add_measurement(m); }
			}, instructions[instructions.size() - i - 1]);
		}

		return qc;
	} else {
		throw std::invalid_argument("Params passed but nothing to bind.");
	}
}

QuantumCircuit QuantumCircuit::reverse() const {
  QuantumCircuit qc(num_qubits);
	for (uint32_t i = 0; i < instructions.size(); i++) {
    std::visit(quantumcircuit_utils::overloaded {
      [&qc](std::shared_ptr<Gate> gate) { qc.add_gate(gate); },
      [&qc](Measurement m) { qc.add_measurement(m); }
    }, instructions[instructions.size() - i - 1]);
  }
  return qc;
}

QuantumCircuit QuantumCircuit::conjugate(const QuantumCircuit& other) const {
  if (num_qubits != other.num_qubits) {
    throw std::runtime_error("Mismatch in number of qubits in QuantumCircuit.conjugate.");
  }
  
  if (num_params() != 0 || other.num_params() != 0) {
    throw std::runtime_error("Unbound parameters, cannot performon QuantumCircuit.conjugate.");
  }

  QuantumCircuit qc(num_qubits);
  qc.append(other);
  qc.append(*this);
  qc.append(other.adjoint());
  return qc;
}

Eigen::MatrixXcd QuantumCircuit::to_matrix(const std::optional<std::vector<double>>& params_opt) const {
  size_t nparams = num_params();
	if (params_opt) { 
		auto params = params_opt.value();
		if (params.size() < nparams) {
			throw std::invalid_argument("Unbound parameters; cannot convert circuit to matrix.");
		} else if (params.size() > nparams) {
      throw std::invalid_argument("Too many parameters passed; cannot convert circuit to matrix.");
    }

		QuantumCircuit qc = bind_params(params);
		return qc.to_matrix();
	} else {
    if (nparams > 0) {
			throw std::invalid_argument("Unbound parameters; cannot convert circuit to matrix.");
    }

    if (num_qubits > 15) {
      throw std::runtime_error("Cannot convert QuantumCircuit with n > 15 qubits to matrix.");
    }

		Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(1u << num_qubits, 1u << num_qubits);
		Q.setIdentity();

		uint32_t p = num_qubits;

		for (uint32_t i = 0; i < instructions.size(); i++) {
			std::visit(quantumcircuit_utils::overloaded {
				[&Q, p](std::shared_ptr<Gate> gate) { Q = full_circuit_unitary(gate->define(), gate->qubits, p) * Q; },
				[](Measurement m) { throw std::invalid_argument("Cannot convert measurement to matrix."); }
			}, instructions[i]);
		}

		return Q;
	}
}

// --- Library for building common circuits --- //
QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc, std::optional<int> seed) {
	thread_local std::random_device rd;
	std::minstd_rand rng;
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
				auto gate = parse_gate(s, Qubits{q1});
				assert(gate->num_qubits == 1);
				circuit.add_gate(gate);
				gate = parse_gate(s, Qubits{q2});
				circuit.add_gate(gate);
			}

			auto entangler = parse_gate(entangling_gate, Qubits{q1, q2});
			assert(entangler->num_qubits == 2);
			circuit.add_gate(entangler);
		}
	}

	if (final_layer) {
		for (uint32_t q = 0; q < num_qubits; q++) {
			for (auto const &s : rotation_gates) {
				auto gate = parse_gate(s, Qubits{q});
				assert(gate->num_qubits == 1);
				circuit.add_gate(gate);
			}
		}
	}

	return circuit;
}

QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<Qubits>& qubits_opt) {
	auto qubits = parse_qargs_opt(qubits_opt, num_qubits);

	QuantumCircuit circuit(num_qubits);
	
	for (auto const& q : qubits) {
		circuit.add_gate(std::make_shared<RxRotationGate>(Qubits{q}));
	}
	
	return circuit;
}

QuantumCircuit random_clifford(uint32_t num_qubits, std::minstd_rand& rng) {
  QuantumCircuit qc(num_qubits);

  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  random_clifford_impl(qubits, rng, qc);

  return qc;
}

