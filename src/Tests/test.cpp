#include "BinaryMatrix.hpp"
#include "BinaryPolynomial.hpp"
#include <iostream>
#include <cassert>

bool test_solve_linear_system() {
  BinaryMatrix M(4, 4);
  M.set(0,0,1);
  M.set(0,1,1);
  M.set(0,2,1);
  M.set(0,3,1);

  M.set(1,0,1);
  M.set(1,1,1);

  M.set(2,0,1);
  M.set(2,2,1);

  M.set(3,0,1);
  std::vector<bool> v{0, 0, 0, 1};
  auto x = M.solve_linear_system(v);
  std::vector<bool> correct{1, 1, 1, 1};

  if (x != correct) {
    return false;
  }
}

bool test_binary_polynomial() {
  BinaryPolynomial poly(5);
  poly.add_term(1, 2);
  poly.add_term(1);
  poly.add_term({});

  std::cout << poly.to_string() << std::endl;

  return true;
}


int main() {
  //assert(test_solve_linear_system());
  test_binary_polynomial();
}
