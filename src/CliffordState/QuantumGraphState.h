#pragma once

#include <vector>
#include "Graph.hpp"
#include "CliffordState.hpp"
#include "QuantumCHPState.hpp"

#include "QuantumState.h"

#define IDGATE     0
#define XGATE      1
#define YGATE      2
#define ZGATE      3
#define HGATE      4
#define SGATE      8
#define SDGATE     11
#define SQRTXGATE  20
#define SQRTXDGATE 21
#define SQRTYGATE  7
#define SQRTYDGATE 5
#define SQRTZGATE  8
#define SQRTZDGATE 11


// A simulator for quantum clifford states represented with graphs, as outlined in
// https://arxiv.org/abs/quant-ph/0504117
class QuantumGraphState : public CliffordState {
  private:
    static const uint32_t ZGATES[4];
    static const uint32_t CONJUGATION_TABLE[24];
    static const uint32_t HERMITIAN_CONJUGATE_TABLE[24];
    static const uint32_t CLIFFORD_DECOMPS[24][5];
    static const uint32_t CLIFFORD_PRODUCTS[24][24];
    static const uint32_t CZ_LOOKUP[24][24][2][3];

    void apply_gater(uint32_t a, uint gate_id);
    void apply_gatel(uint32_t a, uint gate_id);
    void local_complement(uint32_t a);
    void remove_vop(uint32_t a, uint b);
    bool isolated(uint32_t a, uint b);

    void mxr_graph(uint32_t a, bool outcome);
    void myr_graph(uint32_t a, bool outcome);


  public:
    uint32_t num_qubits;
    Graph<> graph;

    QuantumGraphState()=default;
    QuantumGraphState(uint32_t num_qubits);
    QuantumGraphState(Graph<> &graph);

    QuantumCHPState to_chp() const;
    Statevector to_statevector() const;

    virtual std::string to_string() const override;

    virtual void h(uint32_t a) override;
    virtual void s(uint32_t a) override;
    virtual void sd(uint32_t a) override;

    virtual void x(uint32_t a) override;
    virtual void y(uint32_t a) override;
    virtual void z(uint32_t a) override;

    virtual void random_clifford(const std::vector<uint32_t> &qubits) override;

    virtual void cz(uint32_t a, uint b) override;
    virtual double mzr_expectation(uint32_t a) const override;
    virtual bool mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) override;

    void mzr_graph(uint32_t a, bool outcome);

    void toggle_edge_gate(uint32_t a, uint b);

    uint32_t distance(const QuantumGraphState& other) const;

    static double graph_state_entanglement(const Qubits &qubits, Graph<> &graph);
    virtual double entanglement(const QubitSupport &support, uint32_t index) override;

    virtual double sparsity() const override;
    
    std::vector<dataframe::byte_t> serialize() const;
    void deserialize(const std::vector<dataframe::byte_t>& bytes);
};
