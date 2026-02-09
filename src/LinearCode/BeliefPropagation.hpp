#include "LinearCode.h"
#include "Graph.hpp"

using TannerGraph = DirectedGraph<int, double>;

inline size_t variable_idx(size_t i, size_t num_checks) {
  return i + num_checks;
}

inline size_t check_idx(size_t j, size_t num_checks) {
  return j;
}

TannerGraph make_tanner_graph(const ParityCheckMatrix& matrix) {
  uint32_t num_checks = matrix.get_num_rows();
  uint32_t num_variables = matrix.get_num_cols();
  TannerGraph graph(num_variables + num_checks);

  // First num_check variables are check nodes, remaining are variable nodes
  for (uint32_t j = 0; j < num_checks; ++j) {
    for (uint32_t i = 0; i < num_variables; ++i) {
      if (matrix[j][i]) {
        size_t vi = variable_idx(i, num_checks);
        size_t cj = check_idx(j, num_checks);
        graph.add_edge(cj, vi, 0.0);
        graph.add_edge(vi, cj, 0.0);
      }
    }
  }
  
  return graph;
}


std::optional<BitString> belief_propagation(const ParityCheckMatrix& matrix, const BitString& syndrome, const std::vector<double>& error_probs, size_t num_iterations, std::vector<double>& probs) {
  TannerGraph graph = make_tanner_graph(matrix);

  size_t num_checks = matrix.get_num_rows();
  size_t num_variables = matrix.get_num_cols();
  size_t num_nodes = num_checks + num_variables;

  std::vector<double> lch(num_variables);
  for (size_t i = 0; i < num_variables; ++i) {
    lch[i] = std::log((1.0 - error_probs[i])/error_probs[i]);
  }
  
  // Initialization
  for (size_t i = 0; i < num_variables; ++i) {
    size_t vi = variable_idx(i, num_checks);
    for (auto& [cj, mu] : graph.edges[vi]) {
      mu = lch[i];
    }
  }

  std::cout << "M = \n" << matrix.to_string() << "\nG = \n" << graph.to_string() << "\n\n";

  for (size_t t = 0; t < num_iterations; ++t) {
    for (size_t i = 0; i < num_variables; ++i) {
      size_t vi = variable_idx(i, num_checks);
      double mu = lch[i];

      for (const auto& [cj, _] : graph.edges[vi]) {
        for (const auto& [ck, _] : graph.edges[vi]) {
          if (ck == cj) {
            continue;
          }

          mu += graph.edge_weight(ck, vi);
        }

        graph.set_edge_weight(vi, cj, mu);
      }
    }

    for (size_t j = 0; j < num_checks; ++j) {
      size_t cj = check_idx(j, num_checks);
      double p = 1.0;

      for (const auto& [vi, _] : graph.edges[cj]) {
        for (const auto& [vk, _] : graph.edges[cj]) {
          if (vk == vi) {
            continue;
          }
          p *= std::tanh(graph.edge_weight(vk, cj));
        }

        double sign = syndrome.get(j) ? -1 : 1;
        graph.set_edge_weight(cj, vi, sign * 2 * std::atanh(p));
      }
    }

    std::vector<double> llr_ap(num_variables, 0.0);
    for (size_t i = 0; i < num_variables; ++i) {
      llr_ap[i] = lch[i];

      size_t vi = variable_idx(i, num_checks);
      for (const auto& [ck, muk] : graph.edges[vi]) {
        llr_ap[i] += muk;
      }
    }

    probs = llr_ap;
    for (size_t i = 0; i < num_variables; ++i) {
      probs[i] = exp(probs[i]);
    }

    BitString estimated_error(num_variables);
    for (size_t i = 0; i < num_variables; ++i) {
      if (llr_ap[i] < 0.0) {
        estimated_error[i] = true;
      }
    }

    BitString estimated_syndrome = matrix.BinaryMatrixBase::multiply(estimated_error);

    std::cout << fmt::format("On iteration {}, probs = {::.2f}, estimated_error = {}, estimated_syndrome = {}\n", t, llr_ap, estimated_error, estimated_syndrome);

    if (estimated_syndrome == syndrome) {
      return estimated_error;
    }
  }

  return std::nullopt;
}
