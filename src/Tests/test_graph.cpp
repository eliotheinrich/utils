#include "tests.hpp"
#include "Graph.hpp"

bool test_graph() {
  UndirectedGraph<int> g(4);
  g.add_edge(0, 1); 
  g.add_edge(1, 3); // 1 -> 2
  g.add_edge(0, 3);
  g.toggle_edge(2, 3); 
  g.add_edge(0, 2); // 0 -> 2
  g.remove_vertex(2);
  g.remove_edge(0, 2);

  // graph should be 0 <-> 1, 1 <-> 2, and not 0 <-> 2

  ASSERT(g.contains_edge(0, 1));
  ASSERT(g.contains_edge(1, 0));
  ASSERT(g.contains_edge(1, 2));
  ASSERT(g.contains_edge(2, 1));
  ASSERT(!g.contains_edge(0, 2));
  ASSERT(!g.contains_edge(2, 0));
  ASSERT(g.degree(0) == 1);
  ASSERT(g.degree(1) == 2);
  ASSERT(g.degree(2) == 1);

  return true;
}

bool test_directed_graph() {
  DirectedGraph<int> dg(4);
  dg.add_edge(0, 1); 
  dg.add_edge(1, 3); // 1 -> 2
  dg.add_edge(0, 3);
  dg.toggle_edge(2, 3); 
  dg.add_edge(0, 2); // 0 -> 2
  dg.remove_vertex(2);
  dg.remove_edge(0, 2);

  ASSERT(dg.contains_edge(0, 1));
  ASSERT(!dg.contains_edge(1, 0));
  ASSERT(dg.contains_edge(1, 2));
  ASSERT(!dg.contains_edge(2, 1));
  ASSERT(!dg.contains_edge(0, 2));
  ASSERT(!dg.contains_edge(2, 0));
  ASSERT(dg.degree(0) == 1);
  ASSERT(dg.degree(1) == 1);
  ASSERT(dg.degree(2) == 0);

  return true;
}

bool test_max_flow() {
  DirectedGraph<int, int> g(6);
  g.add_edge(0, 1, 11);
  g.add_edge(0, 2, 12);
  g.add_edge(2, 1, 1);
  g.add_edge(1, 3, 12);
  g.add_edge(1, 3, 12);
  g.add_edge(2, 4, 11);
  g.add_edge(4, 3, 7);
  g.add_edge(3, 5, 19);
  g.add_edge(4, 5, 4);

  int flow = g.max_flow({0}, {5});
  ASSERT(flow == 23);
  return true;
}

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }

  ADD_TEST(test_graph);
  ADD_TEST(test_directed_graph);
  ADD_TEST(test_max_flow);

  constexpr char green[] = "\033[1;32m";
  constexpr char black[] = "\033[0m";
  constexpr char red[] = "\033[1;31m";

  auto test_passed_str = [&](bool passed) {
    std::stringstream stream;
    if (passed) {
      stream << green << "PASSED" << black;
    } else {
      stream << red << "FAILED" << black;
    }
    
    return stream.str();
  };

  if (tests.size() == 0) {
    std::cout << "No tests to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, result] : tests) {
      auto [passed, duration] = result;
      std::cout << fmt::format("{:>40}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
