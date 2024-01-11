#include "QuantumGraphState.h"
#include <climits>


const uint32_t QuantumGraphState::ZGATES[4] = {IDGATE, ZGATE, SGATE, SDGATE};

const uint32_t QuantumGraphState::CONJUGATION_TABLE[24] = {3, 6, 6, 3, 1, 1, 4, 4, 3, 6, 6, 3, 5, 2, 5, 2, 1, 1, 4, 4, 2, 5, 2, 5};

// TODO check
const uint32_t QuantumGraphState::HERMITIAN_CONJUGATE_TABLE[24] = {0, 1, 2, 3, 4, 7, 6, 5, 11, 9, 10, 8, 17, 19, 18, 16, 15, 12, 14, 13, 21, 20, 22, 23};

const uint32_t QuantumGraphState::CLIFFORD_DECOMPS[24][5] = 
{{    IDGATE,     IDGATE,     IDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE, SQRTXDGATE,     IDGATE,     IDGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE, SQRTXDGATE, SQRTXDGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE,     IDGATE,     IDGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE, SQRTXDGATE, SQRTXDGATE,  SQRTZGATE},
  { SQRTZGATE,  SQRTZGATE,  SQRTZGATE, SQRTXDGATE,  SQRTZGATE},
  { SQRTZGATE, SQRTXDGATE,  SQRTZGATE,     IDGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE,  SQRTZGATE,  SQRTZGATE,  SQRTZGATE},
  { SQRTZGATE,     IDGATE,     IDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE, SQRTXDGATE,  SQRTZGATE,     IDGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE, SQRTXDGATE,     IDGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE,  SQRTZGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE,  SQRTZGATE,  SQRTZGATE,  SQRTZGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE, SQRTXDGATE,  SQRTZGATE,     IDGATE},
  {SQRTXDGATE,  SQRTZGATE,     IDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE, SQRTXDGATE, SQRTXDGATE,  SQRTZGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE,  SQRTZGATE, SQRTXDGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE, SQRTXDGATE, SQRTXDGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE,  SQRTZGATE,  SQRTZGATE,     IDGATE},
  { SQRTZGATE, SQRTXDGATE,     IDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE, SQRTXDGATE, SQRTXDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE,     IDGATE,     IDGATE,     IDGATE,     IDGATE},
  { SQRTZGATE,  SQRTZGATE, SQRTXDGATE,     IDGATE,     IDGATE},
  {SQRTXDGATE,  SQRTZGATE,  SQRTZGATE,     IDGATE,     IDGATE}};

const uint32_t QuantumGraphState::CLIFFORD_PRODUCTS[24][24] = 
{{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  { 1,  0,  3,  2,  7,  6,  5,  4, 10, 11,  8,  9, 15, 14, 13, 12, 18, 19, 16, 17, 21, 20, 23, 22},
  { 2,  3,  0,  1,  6,  7,  4,  5,  9,  8, 11, 10, 13, 12, 15, 14, 19, 18, 17, 16, 23, 22, 21, 20},
  { 3,  2,  1,  0,  5,  4,  7,  6, 11, 10,  9,  8, 14, 15, 12, 13, 17, 16, 19, 18, 22, 23, 20, 21},
  { 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19},
  { 5,  4,  7,  6,  3,  2,  1,  0, 14, 15, 12, 13, 11, 10,  9,  8, 22, 23, 20, 21, 17, 16, 19, 18},
  { 6,  7,  4,  5,  2,  3,  0,  1, 13, 12, 15, 14,  9,  8, 11, 10, 23, 22, 21, 20, 19, 18, 17, 16},
  { 7,  6,  5,  4,  1,  0,  3,  2, 15, 14, 13, 12, 10, 11,  8,  9, 21, 20, 23, 22, 18, 19, 16, 17},
  { 8,  9, 10, 11, 16, 17, 18, 19,  3,  2,  1,  0, 21, 20, 23, 22,  5,  4,  7,  6, 15, 14, 13, 12},
  { 9,  8, 11, 10, 19, 18, 17, 16,  1,  0,  3,  2, 22, 23, 20, 21,  7,  6,  5,  4, 14, 15, 12, 13},
  {10, 11,  8,  9, 18, 19, 16, 17,  2,  3,  0,  1, 20, 21, 22, 23,  6,  7,  4,  5, 12, 13, 14, 15},
  {11, 10,  9,  8, 17, 16, 19, 18,  0,  1,  2,  3, 23, 22, 21, 20,  4,  5,  6,  7, 13, 12, 15, 14},
  {12, 13, 14, 15, 20, 21, 22, 23,  7,  6,  5,  4, 17, 16, 19, 18,  1,  0,  3,  2, 11, 10,  9,  8},
  {13, 12, 15, 14, 23, 22, 21, 20,  5,  4,  7,  6, 18, 19, 16, 17,  3,  2,  1,  0, 10, 11,  8,  9},
  {14, 15, 12, 13, 22, 23, 20, 21,  6,  7,  4,  5, 16, 17, 18, 19,  2,  3,  0,  1,  8,  9, 10, 11},
  {15, 14, 13, 12, 21, 20, 23, 22,  4,  5,  6,  7, 19, 18, 17, 16,  0,  1,  2,  3,  9,  8, 11, 10},
  {16, 17, 18, 19,  8,  9, 10, 11, 21, 20, 23, 22,  3,  2,  1,  0, 15, 14, 13, 12,  5,  4,  7,  6},
  {17, 16, 19, 18, 11, 10,  9,  8, 23, 22, 21, 20,  0,  1,  2,  3, 13, 12, 15, 14,  4,  5,  6,  7},
  {18, 19, 16, 17, 10, 11,  8,  9, 20, 21, 22, 23,  2,  3,  0,  1, 12, 13, 14, 15,  6,  7,  4,  5},
  {19, 18, 17, 16,  9,  8, 11, 10, 22, 23, 20, 21,  1,  0,  3,  2, 14, 15, 12, 13,  7,  6,  5,  4},
  {20, 21, 22, 23, 12, 13, 14, 15, 17, 16, 19, 18,  7,  6,  5,  4, 11, 10,  9,  8,  1,  0,  3,  2},
  {21, 20, 23, 22, 15, 14, 13, 12, 19, 18, 17, 16,  4,  5,  6,  7,  9,  8, 11, 10,  0,  1,  2,  3},
  {22, 23, 20, 21, 14, 15, 12, 13, 16, 17, 18, 19,  6,  7,  4,  5,  8,  9, 10, 11,  2,  3,  0,  1},
  {23, 22, 21, 20, 13, 12, 15, 14, 18, 19, 16, 17,  5,  4,  7,  6, 10, 11,  8,  9,  3,  2,  1,  0}};

const uint32_t QuantumGraphState::CZ_LOOKUP[24][24][2][3] = 
{{{{ 0,  0,  1}, { 0,  0,  0}}, {{ 0,  0,  1}, { 3,  0,  0}}, {{ 0,  3,  1}, { 3,  2,  0}}, {{ 0,  3,  1}, { 0,  3,  0}}, {{ 0,  4,  0}, { 0,  5,  1}}, {{ 0,  4,  0}, { 0,  4,  1}}, {{ 3,  6,  0}, { 0,  6,  1}}, {{ 3,  6,  0}, { 0,  7,  1}}, {{ 0,  8,  1}, { 0,  8,  0}}, {{ 0,  8,  1}, { 3,  8,  0}}, {{ 0, 11,  1}, { 3, 10,  0}}, {{ 0, 11,  1}, { 0, 11,  0}}, {{ 0, 11,  1}, {11, 10,  0}}, {{ 0, 11,  1}, { 8, 10,  0}}, {{ 0,  8,  1}, {11,  8,  0}}, {{ 0,  8,  1}, { 8,  8,  0}}, {{ 0,  4,  0}, { 0, 17,  1}}, {{ 0,  4,  0}, { 0, 16,  1}}, {{ 3,  6,  0}, { 0, 18,  1}}, {{ 3,  6,  0}, { 0, 19,  1}}, {{ 0,  0,  1}, { 8,  0,  0}}, {{ 0,  0,  1}, {11,  0,  0}}, {{ 0,  3,  1}, { 8,  2,  0}}, {{ 0,  3,  1}, {11,  2,  0}}},
  {{{ 0,  0,  1}, { 0,  3,  0}}, {{ 0,  0,  1}, { 2,  2,  0}}, {{ 0,  3,  1}, { 2,  0,  0}}, {{ 0,  3,  1}, { 0,  0,  0}}, {{ 0,  4,  0}, { 0,  7,  1}}, {{ 0,  4,  0}, { 0,  6,  1}}, {{ 2,  6,  0}, { 0,  4,  1}}, {{ 2,  6,  0}, { 0,  5,  1}}, {{ 0,  8,  1}, { 0, 11,  0}}, {{ 0,  8,  1}, { 2, 10,  0}}, {{ 0, 11,  1}, { 2,  8,  0}}, {{ 0, 11,  1}, { 0,  8,  0}}, {{ 0, 11,  1}, { 8,  8,  0}}, {{ 0, 11,  1}, {10,  8,  0}}, {{ 0,  8,  1}, { 8, 10,  0}}, {{ 0,  8,  1}, {10, 10,  0}}, {{ 0,  4,  0}, { 0, 19,  1}}, {{ 0,  4,  0}, { 0, 18,  1}}, {{ 2,  6,  0}, { 0, 16,  1}}, {{ 2,  6,  0}, { 0, 17,  1}}, {{ 0,  0,  1}, {10,  2,  0}}, {{ 0,  0,  1}, { 8,  2,  0}}, {{ 0,  3,  1}, {10,  0,  0}}, {{ 0,  3,  1}, { 8,  0,  0}}},
  {{{ 2,  3,  1}, { 2,  3,  0}}, {{ 0,  1,  1}, { 0,  2,  0}}, {{ 0,  2,  1}, { 0,  0,  0}}, {{ 2,  0,  1}, { 2,  0,  0}}, {{ 2,  4,  0}, { 0,  6,  1}}, {{ 2,  4,  0}, { 0,  7,  1}}, {{ 0,  6,  0}, { 0,  5,  1}}, {{ 0,  6,  0}, { 0,  4,  1}}, {{ 2, 11,  1}, { 2, 11,  0}}, {{ 0,  9,  1}, { 0, 10,  0}}, {{ 0, 10,  1}, { 0,  8,  0}}, {{ 2,  8,  1}, { 2,  8,  0}}, {{ 0, 10,  1}, {10,  8,  0}}, {{ 0, 10,  1}, { 8,  8,  0}}, {{ 0,  9,  1}, {10, 10,  0}}, {{ 0,  9,  1}, { 8, 10,  0}}, {{ 2,  4,  0}, { 0, 18,  1}}, {{ 2,  4,  0}, { 0, 19,  1}}, {{ 0,  6,  0}, { 0, 17,  1}}, {{ 0,  6,  0}, { 0, 16,  1}}, {{ 0,  1,  1}, { 8,  2,  0}}, {{ 0,  1,  1}, {10,  2,  0}}, {{ 0,  2,  1}, { 8,  0,  0}}, {{ 0,  2,  1}, {10,  0,  0}}},
  {{{ 3,  0,  1}, { 3,  0,  0}}, {{ 0,  1,  1}, { 0,  0,  0}}, {{ 0,  2,  1}, { 0,  2,  0}}, {{ 3,  3,  1}, { 3,  3,  0}}, {{ 3,  4,  0}, { 0,  4,  1}}, {{ 3,  4,  0}, { 0,  5,  1}}, {{ 0,  6,  0}, { 0,  7,  1}}, {{ 0,  6,  0}, { 0,  6,  1}}, {{ 3,  8,  1}, { 3,  8,  0}}, {{ 0,  9,  1}, { 0,  8,  0}}, {{ 0, 10,  1}, { 0, 10,  0}}, {{ 3, 11,  1}, { 3, 11,  0}}, {{ 0, 10,  1}, { 8, 10,  0}}, {{ 0, 10,  1}, {11, 10,  0}}, {{ 0,  9,  1}, { 8,  8,  0}}, {{ 0,  9,  1}, {11,  8,  0}}, {{ 3,  4,  0}, { 0, 16,  1}}, {{ 3,  4,  0}, { 0, 17,  1}}, {{ 0,  6,  0}, { 0, 19,  1}}, {{ 0,  6,  0}, { 0, 18,  1}}, {{ 0,  1,  1}, {11,  0,  0}}, {{ 0,  1,  1}, { 8,  0,  0}}, {{ 0,  2,  1}, {11,  2,  0}}, {{ 0,  2,  1}, { 8,  2,  0}}},
  {{{ 4,  0,  0}, { 4,  3,  1}}, {{ 4,  0,  0}, { 0,  7,  1}}, {{ 4,  2,  0}, { 0,  6,  1}}, {{ 4,  3,  0}, { 4,  0,  1}}, {{ 4,  4,  0}, { 0,  0,  0}}, {{ 4,  4,  0}, { 0,  2,  0}}, {{ 4,  6,  0}, { 2,  2,  0}}, {{ 4,  6,  0}, { 2,  0,  0}}, {{ 4,  8,  0}, { 4, 11,  1}}, {{ 4,  8,  0}, { 0, 19,  1}}, {{ 4, 10,  0}, { 0, 18,  1}}, {{ 4, 11,  0}, { 4,  8,  1}}, {{ 4, 10,  0}, { 8,  0,  0}}, {{ 4, 10,  0}, {10,  2,  0}}, {{ 4,  8,  0}, { 8,  2,  0}}, {{ 4,  8,  0}, {10,  0,  0}}, {{ 4,  4,  0}, { 0,  8,  0}}, {{ 4,  4,  0}, { 0, 10,  0}}, {{ 4,  6,  0}, { 2, 10,  0}}, {{ 4,  6,  0}, { 2,  8,  0}}, {{ 4,  0,  0}, {10, 10,  0}}, {{ 4,  0,  0}, { 8,  8,  0}}, {{ 4,  2,  0}, {10,  8,  0}}, {{ 4,  2,  0}, { 8, 10,  0}}},
  {{{ 4,  0,  0}, { 4,  0,  1}}, {{ 4,  0,  0}, { 0,  6,  1}}, {{ 4,  2,  0}, { 0,  7,  1}}, {{ 4,  3,  0}, { 4,  3,  1}}, {{ 4,  4,  0}, { 2,  0,  0}}, {{ 4,  4,  0}, { 2,  2,  0}}, {{ 4,  6,  0}, { 0,  2,  0}}, {{ 4,  6,  0}, { 0,  0,  0}}, {{ 4,  8,  0}, { 4,  8,  1}}, {{ 4,  8,  0}, { 0, 18,  1}}, {{ 4, 10,  0}, { 0, 19,  1}}, {{ 4, 11,  0}, { 4, 11,  1}}, {{ 4, 10,  0}, {10,  0,  0}}, {{ 4, 10,  0}, { 8,  2,  0}}, {{ 4,  8,  0}, {10,  2,  0}}, {{ 4,  8,  0}, { 8,  0,  0}}, {{ 4,  4,  0}, { 2,  8,  0}}, {{ 4,  4,  0}, { 2, 10,  0}}, {{ 4,  6,  0}, { 0, 10,  0}}, {{ 4,  6,  0}, { 0,  8,  0}}, {{ 4,  0,  0}, { 8, 10,  0}}, {{ 4,  0,  0}, {10,  8,  0}}, {{ 4,  2,  0}, { 8,  8,  0}}, {{ 4,  2,  0}, {10, 10,  0}}},
  {{{ 6,  3,  0}, { 6,  0,  1}}, {{ 6,  2,  0}, { 0,  4,  1}}, {{ 6,  0,  0}, { 0,  5,  1}}, {{ 6,  0,  0}, { 6,  3,  1}}, {{ 6,  4,  0}, { 2,  2,  0}}, {{ 6,  4,  0}, { 2,  0,  0}}, {{ 6,  6,  0}, { 0,  0,  0}}, {{ 6,  6,  0}, { 0,  2,  0}}, {{ 6, 11,  0}, { 6,  8,  1}}, {{ 6, 10,  0}, { 0, 16,  1}}, {{ 6,  8,  0}, { 0, 17,  1}}, {{ 6,  8,  0}, { 6, 11,  1}}, {{ 6,  8,  0}, { 8,  2,  0}}, {{ 6,  8,  0}, {10,  0,  0}}, {{ 6, 10,  0}, { 8,  0,  0}}, {{ 6, 10,  0}, {10,  2,  0}}, {{ 6,  4,  0}, { 2, 10,  0}}, {{ 6,  4,  0}, { 2,  8,  0}}, {{ 6,  6,  0}, { 0,  8,  0}}, {{ 6,  6,  0}, { 0, 10,  0}}, {{ 6,  2,  0}, {10,  8,  0}}, {{ 6,  2,  0}, { 8, 10,  0}}, {{ 6,  0,  0}, {10, 10,  0}}, {{ 6,  0,  0}, { 8,  8,  0}}},
  {{{ 6,  3,  0}, { 6,  3,  1}}, {{ 6,  2,  0}, { 0,  5,  1}}, {{ 6,  0,  0}, { 0,  4,  1}}, {{ 6,  0,  0}, { 6,  0,  1}}, {{ 6,  4,  0}, { 0,  2,  0}}, {{ 6,  4,  0}, { 0,  0,  0}}, {{ 6,  6,  0}, { 2,  0,  0}}, {{ 6,  6,  0}, { 2,  2,  0}}, {{ 6, 11,  0}, { 6, 11,  1}}, {{ 6, 10,  0}, { 0, 17,  1}}, {{ 6,  8,  0}, { 0, 16,  1}}, {{ 6,  8,  0}, { 6,  8,  1}}, {{ 6,  8,  0}, {10,  2,  0}}, {{ 6,  8,  0}, { 8,  0,  0}}, {{ 6, 10,  0}, {10,  0,  0}}, {{ 6, 10,  0}, { 8,  2,  0}}, {{ 6,  4,  0}, { 0, 10,  0}}, {{ 6,  4,  0}, { 0,  8,  0}}, {{ 6,  6,  0}, { 2,  8,  0}}, {{ 6,  6,  0}, { 2, 10,  0}}, {{ 6,  2,  0}, { 8,  8,  0}}, {{ 6,  2,  0}, {10, 10,  0}}, {{ 6,  0,  0}, { 8, 10,  0}}, {{ 6,  0,  0}, {10,  8,  0}}},
  {{{ 8,  0,  1}, { 8,  0,  0}}, {{ 0, 20,  1}, {11,  0,  0}}, {{ 0, 22,  1}, {11,  2,  0}}, {{ 8,  3,  1}, { 8,  3,  0}}, {{ 8,  4,  0}, { 0, 17,  1}}, {{ 8,  4,  0}, { 0, 16,  1}}, {{11,  6,  0}, { 0, 19,  1}}, {{11,  6,  0}, { 0, 18,  1}}, {{ 8,  8,  1}, { 8,  8,  0}}, {{ 0, 15,  1}, {11,  8,  0}}, {{ 0, 13,  1}, {11, 10,  0}}, {{ 8, 11,  1}, { 8, 11,  0}}, {{ 0, 13,  1}, { 0, 10,  0}}, {{ 0, 13,  1}, { 3, 10,  0}}, {{ 0, 15,  1}, { 0,  8,  0}}, {{ 0, 15,  1}, { 3,  8,  0}}, {{ 8,  4,  0}, { 0,  4,  1}}, {{ 8,  4,  0}, { 0,  5,  1}}, {{11,  6,  0}, { 0,  6,  1}}, {{11,  6,  0}, { 0,  7,  1}}, {{ 0, 20,  1}, { 3,  0,  0}}, {{ 0, 20,  1}, { 0,  0,  0}}, {{ 0, 22,  1}, { 3,  2,  0}}, {{ 0, 22,  1}, { 0,  2,  0}}},
  {{{ 8,  0,  1}, { 8,  3,  0}}, {{ 0, 20,  1}, {10,  2,  0}}, {{ 0, 22,  1}, {10,  0,  0}}, {{ 8,  3,  1}, { 8,  0,  0}}, {{ 8,  4,  0}, { 0, 18,  1}}, {{ 8,  4,  0}, { 0, 19,  1}}, {{10,  6,  0}, { 0, 16,  1}}, {{10,  6,  0}, { 0, 17,  1}}, {{ 8,  8,  1}, { 8, 11,  0}}, {{ 0, 15,  1}, {10, 10,  0}}, {{ 0, 13,  1}, {10,  8,  0}}, {{ 8, 11,  1}, { 8,  8,  0}}, {{ 0, 13,  1}, { 2,  8,  0}}, {{ 0, 13,  1}, { 0,  8,  0}}, {{ 0, 15,  1}, { 2, 10,  0}}, {{ 0, 15,  1}, { 0, 10,  0}}, {{ 8,  4,  0}, { 0,  7,  1}}, {{ 8,  4,  0}, { 0,  6,  1}}, {{10,  6,  0}, { 0,  5,  1}}, {{10,  6,  0}, { 0,  4,  1}}, {{ 0, 20,  1}, { 0,  2,  0}}, {{ 0, 20,  1}, { 2,  2,  0}}, {{ 0, 22,  1}, { 0,  0,  0}}, {{ 0, 22,  1}, { 2,  0,  0}}},
  {{{10,  3,  1}, {10,  3,  0}}, {{ 0, 21,  1}, { 8,  2,  0}}, {{ 0, 23,  1}, { 8,  0,  0}}, {{10,  0,  1}, {10,  0,  0}}, {{10,  4,  0}, { 0, 19,  1}}, {{10,  4,  0}, { 0, 18,  1}}, {{ 8,  6,  0}, { 0, 17,  1}}, {{ 8,  6,  0}, { 0, 16,  1}}, {{10, 11,  1}, {10, 11,  0}}, {{ 0, 14,  1}, { 8, 10,  0}}, {{ 0, 12,  1}, { 8,  8,  0}}, {{10,  8,  1}, {10,  8,  0}}, {{ 0, 12,  1}, { 0,  8,  0}}, {{ 0, 12,  1}, { 2,  8,  0}}, {{ 0, 14,  1}, { 0, 10,  0}}, {{ 0, 14,  1}, { 2, 10,  0}}, {{10,  4,  0}, { 0,  6,  1}}, {{10,  4,  0}, { 0,  7,  1}}, {{ 8,  6,  0}, { 0,  4,  1}}, {{ 8,  6,  0}, { 0,  5,  1}}, {{ 0, 21,  1}, { 2,  2,  0}}, {{ 0, 21,  1}, { 0,  2,  0}}, {{ 0, 23,  1}, { 2,  0,  0}}, {{ 0, 23,  1}, { 0,  0,  0}}},
  {{{11,  0,  1}, {11,  0,  0}}, {{ 0, 21,  1}, { 8,  0,  0}}, {{ 0, 23,  1}, { 8,  2,  0}}, {{11,  3,  1}, {11,  3,  0}}, {{11,  4,  0}, { 0, 16,  1}}, {{11,  4,  0}, { 0, 17,  1}}, {{ 8,  6,  0}, { 0, 18,  1}}, {{ 8,  6,  0}, { 0, 19,  1}}, {{11,  8,  1}, {11,  8,  0}}, {{ 0, 14,  1}, { 8,  8,  0}}, {{ 0, 12,  1}, { 8, 10,  0}}, {{11, 11,  1}, {11, 11,  0}}, {{ 0, 12,  1}, { 3, 10,  0}}, {{ 0, 12,  1}, { 0, 10,  0}}, {{ 0, 14,  1}, { 3,  8,  0}}, {{ 0, 14,  1}, { 0,  8,  0}}, {{11,  4,  0}, { 0,  5,  1}}, {{11,  4,  0}, { 0,  4,  1}}, {{ 8,  6,  0}, { 0,  7,  1}}, {{ 8,  6,  0}, { 0,  6,  1}}, {{ 0, 21,  1}, { 0,  0,  0}}, {{ 0, 21,  1}, { 3,  0,  0}}, {{ 0, 23,  1}, { 0,  2,  0}}, {{ 0, 23,  1}, { 3,  2,  0}}},
  {{{10,  3,  1}, {10, 11,  0}}, {{ 0, 21,  1}, { 8,  8,  0}}, {{ 0, 23,  1}, { 8, 10,  0}}, {{10,  0,  1}, {10,  8,  0}}, {{10,  4,  0}, { 0,  8,  0}}, {{10,  4,  0}, { 0, 10,  0}}, {{ 8,  6,  0}, { 2,  8,  0}}, {{ 8,  6,  0}, { 2, 10,  0}}, {{10, 11,  1}, {10,  0,  0}}, {{ 0, 14,  1}, { 8,  2,  0}}, {{ 0, 12,  1}, { 8,  0,  0}}, {{10,  8,  1}, {10,  3,  0}}, {{ 0, 12,  1}, { 0, 16,  1}}, {{ 0, 12,  1}, { 0, 18,  1}}, {{ 0, 14,  1}, { 0, 17,  1}}, {{ 0, 14,  1}, { 0, 19,  1}}, {{10,  4,  0}, { 0,  2,  0}}, {{10,  4,  0}, { 0,  0,  0}}, {{ 8,  6,  0}, { 2,  2,  0}}, {{ 8,  6,  0}, { 2,  0,  0}}, {{ 0, 21,  1}, { 0,  7,  1}}, {{ 0, 21,  1}, { 0,  5,  1}}, {{ 0, 23,  1}, { 0,  6,  1}}, {{ 0, 23,  1}, { 0,  4,  1}}},
  {{{10,  3,  1}, {10,  8,  0}}, {{ 0, 21,  1}, { 8, 10,  0}}, {{ 0, 23,  1}, { 8,  8,  0}}, {{10,  0,  1}, {10, 11,  0}}, {{10,  4,  0}, { 2, 10,  0}}, {{10,  4,  0}, { 2,  8,  0}}, {{ 8,  6,  0}, { 0, 10,  0}}, {{ 8,  6,  0}, { 0,  8,  0}}, {{10, 11,  1}, {10,  3,  0}}, {{ 0, 14,  1}, { 8,  0,  0}}, {{ 0, 12,  1}, { 8,  2,  0}}, {{10,  8,  1}, {10,  0,  0}}, {{ 0, 12,  1}, { 0, 19,  1}}, {{ 0, 12,  1}, { 0, 17,  1}}, {{ 0, 14,  1}, { 0, 18,  1}}, {{ 0, 14,  1}, { 0, 16,  1}}, {{10,  4,  0}, { 2,  0,  0}}, {{10,  4,  0}, { 2,  2,  0}}, {{ 8,  6,  0}, { 0,  0,  0}}, {{ 8,  6,  0}, { 0,  2,  0}}, {{ 0, 21,  1}, { 0,  4,  1}}, {{ 0, 21,  1}, { 0,  6,  1}}, {{ 0, 23,  1}, { 0,  5,  1}}, {{ 0, 23,  1}, { 0,  7,  1}}},
  {{{ 8,  0,  1}, { 8, 11,  0}}, {{ 0, 20,  1}, {10,  8,  0}}, {{ 0, 22,  1}, {10, 10,  0}}, {{ 8,  3,  1}, { 8,  8,  0}}, {{ 8,  4,  0}, { 2,  8,  0}}, {{ 8,  4,  0}, { 2, 10,  0}}, {{10,  6,  0}, { 0,  8,  0}}, {{10,  6,  0}, { 0, 10,  0}}, {{ 8,  8,  1}, { 8,  0,  0}}, {{ 0, 15,  1}, {10,  2,  0}}, {{ 0, 13,  1}, {10,  0,  0}}, {{ 8, 11,  1}, { 8,  3,  0}}, {{ 0, 13,  1}, { 0, 17,  1}}, {{ 0, 13,  1}, { 0, 19,  1}}, {{ 0, 15,  1}, { 0, 16,  1}}, {{ 0, 15,  1}, { 0, 18,  1}}, {{ 8,  4,  0}, { 2,  2,  0}}, {{ 8,  4,  0}, { 2,  0,  0}}, {{10,  6,  0}, { 0,  2,  0}}, {{10,  6,  0}, { 0,  0,  0}}, {{ 0, 20,  1}, { 0,  6,  1}}, {{ 0, 20,  1}, { 0,  4,  1}}, {{ 0, 22,  1}, { 0,  7,  1}}, {{ 0, 22,  1}, { 0,  5,  1}}},
  {{{ 8,  0,  1}, { 8,  8,  0}}, {{ 0, 20,  1}, {10, 10,  0}}, {{ 0, 22,  1}, {10,  8,  0}}, {{ 8,  3,  1}, { 8, 11,  0}}, {{ 8,  4,  0}, { 0, 10,  0}}, {{ 8,  4,  0}, { 0,  8,  0}}, {{10,  6,  0}, { 2, 10,  0}}, {{10,  6,  0}, { 2,  8,  0}}, {{ 8,  8,  1}, { 8,  3,  0}}, {{ 0, 15,  1}, {10,  0,  0}}, {{ 0, 13,  1}, {10,  2,  0}}, {{ 8, 11,  1}, { 8,  0,  0}}, {{ 0, 13,  1}, { 0, 18,  1}}, {{ 0, 13,  1}, { 0, 16,  1}}, {{ 0, 15,  1}, { 0, 19,  1}}, {{ 0, 15,  1}, { 0, 17,  1}}, {{ 8,  4,  0}, { 0,  0,  0}}, {{ 8,  4,  0}, { 0,  2,  0}}, {{10,  6,  0}, { 2,  0,  0}}, {{10,  6,  0}, { 2,  2,  0}}, {{ 0, 20,  1}, { 0,  5,  1}}, {{ 0, 20,  1}, { 0,  7,  1}}, {{ 0, 22,  1}, { 0,  4,  1}}, {{ 0, 22,  1}, { 0,  6,  1}}},
  {{{ 4,  0,  0}, { 4, 11,  1}}, {{ 4,  0,  0}, { 0, 18,  1}}, {{ 4,  2,  0}, { 0, 19,  1}}, {{ 4,  3,  0}, { 4,  8,  1}}, {{ 4,  4,  0}, { 8,  0,  0}}, {{ 4,  4,  0}, { 8,  2,  0}}, {{ 4,  6,  0}, {10,  2,  0}}, {{ 4,  6,  0}, {10,  0,  0}}, {{ 4,  8,  0}, { 4,  0,  1}}, {{ 4,  8,  0}, { 0,  7,  1}}, {{ 4, 10,  0}, { 0,  6,  1}}, {{ 4, 11,  0}, { 4,  3,  1}}, {{ 4, 10,  0}, { 2,  0,  0}}, {{ 4, 10,  0}, { 0,  2,  0}}, {{ 4,  8,  0}, { 2,  2,  0}}, {{ 4,  8,  0}, { 0,  0,  0}}, {{ 4,  4,  0}, { 8,  8,  0}}, {{ 4,  4,  0}, { 8, 10,  0}}, {{ 4,  6,  0}, {10, 10,  0}}, {{ 4,  6,  0}, {10,  8,  0}}, {{ 4,  0,  0}, { 0, 10,  0}}, {{ 4,  0,  0}, { 2,  8,  0}}, {{ 4,  2,  0}, { 0,  8,  0}}, {{ 4,  2,  0}, { 2, 10,  0}}},
  {{{ 4,  0,  0}, { 4,  8,  1}}, {{ 4,  0,  0}, { 0, 19,  1}}, {{ 4,  2,  0}, { 0, 18,  1}}, {{ 4,  3,  0}, { 4, 11,  1}}, {{ 4,  4,  0}, {10,  0,  0}}, {{ 4,  4,  0}, {10,  2,  0}}, {{ 4,  6,  0}, { 8,  2,  0}}, {{ 4,  6,  0}, { 8,  0,  0}}, {{ 4,  8,  0}, { 4,  3,  1}}, {{ 4,  8,  0}, { 0,  6,  1}}, {{ 4, 10,  0}, { 0,  7,  1}}, {{ 4, 11,  0}, { 4,  0,  1}}, {{ 4, 10,  0}, { 0,  0,  0}}, {{ 4, 10,  0}, { 2,  2,  0}}, {{ 4,  8,  0}, { 0,  2,  0}}, {{ 4,  8,  0}, { 2,  0,  0}}, {{ 4,  4,  0}, {10,  8,  0}}, {{ 4,  4,  0}, {10, 10,  0}}, {{ 4,  6,  0}, { 8, 10,  0}}, {{ 4,  6,  0}, { 8,  8,  0}}, {{ 4,  0,  0}, { 2, 10,  0}}, {{ 4,  0,  0}, { 0,  8,  0}}, {{ 4,  2,  0}, { 2,  8,  0}}, {{ 4,  2,  0}, { 0, 10,  0}}},
  {{{ 6,  3,  0}, { 6, 11,  1}}, {{ 6,  2,  0}, { 0, 16,  1}}, {{ 6,  0,  0}, { 0, 17,  1}}, {{ 6,  0,  0}, { 6,  8,  1}}, {{ 6,  4,  0}, {10,  2,  0}}, {{ 6,  4,  0}, {10,  0,  0}}, {{ 6,  6,  0}, { 8,  0,  0}}, {{ 6,  6,  0}, { 8,  2,  0}}, {{ 6, 11,  0}, { 6,  0,  1}}, {{ 6, 10,  0}, { 0,  5,  1}}, {{ 6,  8,  0}, { 0,  4,  1}}, {{ 6,  8,  0}, { 6,  3,  1}}, {{ 6,  8,  0}, { 2,  2,  0}}, {{ 6,  8,  0}, { 0,  0,  0}}, {{ 6, 10,  0}, { 2,  0,  0}}, {{ 6, 10,  0}, { 0,  2,  0}}, {{ 6,  4,  0}, {10, 10,  0}}, {{ 6,  4,  0}, {10,  8,  0}}, {{ 6,  6,  0}, { 8,  8,  0}}, {{ 6,  6,  0}, { 8, 10,  0}}, {{ 6,  2,  0}, { 0,  8,  0}}, {{ 6,  2,  0}, { 2, 10,  0}}, {{ 6,  0,  0}, { 0, 10,  0}}, {{ 6,  0,  0}, { 2,  8,  0}}},
  {{{ 6,  3,  0}, { 6,  8,  1}}, {{ 6,  2,  0}, { 0, 17,  1}}, {{ 6,  0,  0}, { 0, 16,  1}}, {{ 6,  0,  0}, { 6, 11,  1}}, {{ 6,  4,  0}, { 8,  2,  0}}, {{ 6,  4,  0}, { 8,  0,  0}}, {{ 6,  6,  0}, {10,  0,  0}}, {{ 6,  6,  0}, {10,  2,  0}}, {{ 6, 11,  0}, { 6,  3,  1}}, {{ 6, 10,  0}, { 0,  4,  1}}, {{ 6,  8,  0}, { 0,  5,  1}}, {{ 6,  8,  0}, { 6,  0,  1}}, {{ 6,  8,  0}, { 0,  2,  0}}, {{ 6,  8,  0}, { 2,  0,  0}}, {{ 6, 10,  0}, { 0,  0,  0}}, {{ 6, 10,  0}, { 2,  2,  0}}, {{ 6,  4,  0}, { 8, 10,  0}}, {{ 6,  4,  0}, { 8,  8,  0}}, {{ 6,  6,  0}, {10,  8,  0}}, {{ 6,  6,  0}, {10, 10,  0}}, {{ 6,  2,  0}, { 2,  8,  0}}, {{ 6,  2,  0}, { 0, 10,  0}}, {{ 6,  0,  0}, { 2, 10,  0}}, {{ 6,  0,  0}, { 0,  8,  0}}},
  {{{ 0,  0,  1}, { 0,  8,  0}}, {{ 0,  0,  1}, { 2, 10,  0}}, {{ 0,  3,  1}, { 2,  8,  0}}, {{ 0,  3,  1}, { 0, 11,  0}}, {{ 0,  4,  0}, {10, 10,  0}}, {{ 0,  4,  0}, {10,  8,  0}}, {{ 2,  6,  0}, { 8, 10,  0}}, {{ 2,  6,  0}, { 8,  8,  0}}, {{ 0,  8,  1}, { 0,  3,  0}}, {{ 0,  8,  1}, { 2,  0,  0}}, {{ 0, 11,  1}, { 2,  2,  0}}, {{ 0, 11,  1}, { 0,  0,  0}}, {{ 0, 11,  1}, { 0,  7,  1}}, {{ 0, 11,  1}, { 0,  4,  1}}, {{ 0,  8,  1}, { 0,  6,  1}}, {{ 0,  8,  1}, { 0,  5,  1}}, {{ 0,  4,  0}, {10,  0,  0}}, {{ 0,  4,  0}, {10,  2,  0}}, {{ 2,  6,  0}, { 8,  0,  0}}, {{ 2,  6,  0}, { 8,  2,  0}}, {{ 0,  0,  1}, { 0, 16,  1}}, {{ 0,  0,  1}, { 0, 19,  1}}, {{ 0,  3,  1}, { 0, 17,  1}}, {{ 0,  3,  1}, { 0, 18,  1}}},
  {{{ 0,  0,  1}, { 0, 11,  0}}, {{ 0,  0,  1}, { 2,  8,  0}}, {{ 0,  3,  1}, { 2, 10,  0}}, {{ 0,  3,  1}, { 0,  8,  0}}, {{ 0,  4,  0}, { 8,  8,  0}}, {{ 0,  4,  0}, { 8, 10,  0}}, {{ 2,  6,  0}, {10,  8,  0}}, {{ 2,  6,  0}, {10, 10,  0}}, {{ 0,  8,  1}, { 0,  0,  0}}, {{ 0,  8,  1}, { 2,  2,  0}}, {{ 0, 11,  1}, { 2,  0,  0}}, {{ 0, 11,  1}, { 0,  3,  0}}, {{ 0, 11,  1}, { 0,  5,  1}}, {{ 0, 11,  1}, { 0,  6,  1}}, {{ 0,  8,  1}, { 0,  4,  1}}, {{ 0,  8,  1}, { 0,  7,  1}}, {{ 0,  4,  0}, { 8,  2,  0}}, {{ 0,  4,  0}, { 8,  0,  0}}, {{ 2,  6,  0}, {10,  2,  0}}, {{ 2,  6,  0}, {10,  0,  0}}, {{ 0,  0,  1}, { 0, 18,  1}}, {{ 0,  0,  1}, { 0, 17,  1}}, {{ 0,  3,  1}, { 0, 19,  1}}, {{ 0,  3,  1}, { 0, 16,  1}}},
  {{{ 2,  3,  1}, { 2,  8,  0}}, {{ 0,  1,  1}, { 0, 10,  0}}, {{ 0,  2,  1}, { 0,  8,  0}}, {{ 2,  0,  1}, { 2, 11,  0}}, {{ 2,  4,  0}, { 8, 10,  0}}, {{ 2,  4,  0}, { 8,  8,  0}}, {{ 0,  6,  0}, {10, 10,  0}}, {{ 0,  6,  0}, {10,  8,  0}}, {{ 2, 11,  1}, { 2,  3,  0}}, {{ 0,  9,  1}, { 0,  0,  0}}, {{ 0, 10,  1}, { 0,  2,  0}}, {{ 2,  8,  1}, { 2,  0,  0}}, {{ 0, 10,  1}, { 0,  6,  1}}, {{ 0, 10,  1}, { 0,  5,  1}}, {{ 0,  9,  1}, { 0,  7,  1}}, {{ 0,  9,  1}, { 0,  4,  1}}, {{ 2,  4,  0}, { 8,  0,  0}}, {{ 2,  4,  0}, { 8,  2,  0}}, {{ 0,  6,  0}, {10,  0,  0}}, {{ 0,  6,  0}, {10,  2,  0}}, {{ 0,  1,  1}, { 0, 17,  1}}, {{ 0,  1,  1}, { 0, 18,  1}}, {{ 0,  2,  1}, { 0, 16,  1}}, {{ 0,  2,  1}, { 0, 19,  1}}},
  {{{ 2,  3,  1}, { 2, 11,  0}}, {{ 0,  1,  1}, { 0,  8,  0}}, {{ 0,  2,  1}, { 0, 10,  0}}, {{ 2,  0,  1}, { 2,  8,  0}}, {{ 2,  4,  0}, {10,  8,  0}}, {{ 2,  4,  0}, {10, 10,  0}}, {{ 0,  6,  0}, { 8,  8,  0}}, {{ 0,  6,  0}, { 8, 10,  0}}, {{ 2, 11,  1}, { 2,  0,  0}}, {{ 0,  9,  1}, { 0,  2,  0}}, {{ 0, 10,  1}, { 0,  0,  0}}, {{ 2,  8,  1}, { 2,  3,  0}}, {{ 0, 10,  1}, { 0,  4,  1}}, {{ 0, 10,  1}, { 0,  7,  1}}, {{ 0,  9,  1}, { 0,  5,  1}}, {{ 0,  9,  1}, { 0,  6,  1}}, {{ 2,  4,  0}, {10,  2,  0}}, {{ 2,  4,  0}, {10,  0,  0}}, {{ 0,  6,  0}, { 8,  2,  0}}, {{ 0,  6,  0}, { 8,  0,  0}}, {{ 0,  1,  1}, { 0, 19,  1}}, {{ 0,  1,  1}, { 0, 16,  1}}, {{ 0,  2,  1}, { 0, 18,  1}}, {{ 0,  2,  1}, { 0, 17,  1}}}};


QuantumGraphState::QuantumGraphState(uint32_t num_qubits, int seed) : CliffordState(num_qubits, seed), num_qubits(num_qubits) {
  graph = Graph<>();
  for (uint32_t i = 0; i < num_qubits; i++) graph.add_vertex(HGATE);
}

QuantumGraphState::QuantumGraphState(Graph<> &graph, int seed) : CliffordState(graph.num_vertices, seed) {
  this->graph = Graph<>(graph);
}

QuantumCHPState QuantumGraphState::to_chp() const {
  uint32_t num_qubits = graph.num_vertices;

  QuantumCHPState chp(num_qubits);

  // Prepare |+...+>
  for (uint32_t i = 0; i < num_qubits; i++)
    chp.h(i);

  // Apply graph edges
  for (uint32_t i = 0; i < num_qubits; i++) {
    for (auto j : graph.neighbors(i)) {
      if (i < j)
        chp.cz(i, j);
    }
  }

  // Apply VOP
  for (uint32_t i = 0; i < num_qubits; i++) {
    uint32_t v = graph.get_val(i);
    for (uint32_t j = 0; j < 5; j++) {
      uint32_t vj = CLIFFORD_DECOMPS[v][j];
      if (vj == SQRTXDGATE) {
        // (-iX)^(1/2) = SHS
        chp.s(i);
        chp.h(i);
        chp.s(i);
      } else if (vj == SQRTZGATE) {
        chp.s(i);
      }
    }
  }

  return chp;
}

Statevector QuantumGraphState::to_statevector() const {
  Eigen::Matrix2cd H; H << 1, 1, 1, -1; H = H/std::sqrt(2);
  Eigen::Matrix2cd sqrtmX; sqrtmX << 1 - 1j, 1 + 1j, 1 + 1j, 1 - 1j; sqrtmX = sqrtmX/2;
  Eigen::Matrix2cd sqrtZ; sqrtZ << 1, 0, 0, 1j;
  Eigen::Matrix4cd CZ; CZ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1;

  uint32_t num_qubits = system_size();

  Statevector state(num_qubits);
  for (uint32_t i = 0; i < num_qubits; i++)
    state.QuantumState::evolve(static_cast<Eigen::MatrixXcd>(H), i);

  for (uint32_t i = 0; i < num_qubits; i++) {
    for (auto j : graph.neighbors(i)) {
      if (j < i) {
        std::vector<uint32_t> qbits{i,j};
        state.evolve(static_cast<Eigen::MatrixXcd>(CZ), qbits);
      }
    }
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    uint32_t v = graph.get_val(i);
    for (uint32_t j = 0; j < 5; j++) {
      uint32_t vj = CLIFFORD_DECOMPS[v][j];
      if (vj == SQRTXDGATE) {
        state.QuantumState::evolve(static_cast<Eigen::MatrixXcd>(sqrtmX), i);
      } else if (vj == SQRTZGATE) {
        state.QuantumState::evolve(static_cast<Eigen::MatrixXcd>(sqrtZ), i);
      }
    }
  }

  return state;
}


void QuantumGraphState::apply_gatel(uint32_t a, uint32_t gate_id) {
  auto r = CLIFFORD_PRODUCTS[gate_id][graph.get_val(a)];
  graph.set_val(a, r);
}

void QuantumGraphState::apply_gater(uint32_t a, uint32_t gate_id) {
  graph.set_val(a, CLIFFORD_PRODUCTS[graph.get_val(a)][gate_id]);
}

void QuantumGraphState::local_complement(uint32_t a) {
  graph.local_complement(a);

  apply_gater(a, SQRTXGATE);
  for (auto e : graph.neighbors(a)) {
    apply_gater(e, SQRTZDGATE);
  }
}

void QuantumGraphState::remove_vop(uint32_t a, uint32_t b) {
  uint32_t vop_decomp[5];
  for (uint32_t i = 0; i < 5; i ++) {
    vop_decomp[i] = CLIFFORD_DECOMPS[graph.get_val(a)][i];
  }

  uint32_t c = b;

  for (auto e : graph.neighbors(a)) {
    if (e != b) {
      c = e;
      break;
    }
  }

  for (auto op : vop_decomp) {
    //for (uint32_t i = 0; i < 5; i++) {
    //	auto op = vop_decomp[5 - i - 1];
    if (op == SQRTXDGATE) {
      local_complement(a);
    } else if (op == SQRTZGATE) {
      local_complement(c);
    }
  }
  }

  bool QuantumGraphState::isolated(uint32_t a, uint32_t b) {
    uint32_t deg = graph.degree(a);

    if (deg == 0) {
      return true;
    }

    if (deg == 1) {
      return graph.contains_edge(a, b);
    }

    return false;
  }

  void QuantumGraphState::mxr_graph(uint32_t a, bool outcome) {
    uint32_t b = graph.neighbors(a)[0];

    std::vector<uint32_t> ngbh_a = graph.neighbors(a);
    std::vector<uint32_t> ngbh_b = graph.neighbors(b);

    if (outcome) {
      apply_gater(a, ZGATE);
      for (auto n : ngbh_b) {
        if ((n != a) && !std::binary_search(ngbh_a.begin(), ngbh_a.end(), n)) {
          apply_gater(n, ZGATE);
        }
      }
      apply_gater(b, SQRTYGATE);
    } else {
      for (auto n : ngbh_a) {
        if ((n != b) && !std::binary_search(ngbh_b.begin(), ngbh_b.end(), n)) {
          apply_gater(n, ZGATE);
        }
      }
      apply_gater(b, SQRTYDGATE);
    }

    for (auto c : ngbh_a) {
      for (auto d : ngbh_b) {
        graph.toggle_edge(c, d);
      }
    }

    for (auto c : ngbh_a) {
      if (std::binary_search(ngbh_b.begin(), ngbh_b.end(), c)) {
        for (auto d : ngbh_a) {
          if (std::binary_search(ngbh_b.begin(), ngbh_b.end(), d)) {
            graph.toggle_edge(c, d);
          }
        }
      }
    }

    for (auto d : ngbh_a) {
      if (d != b) {
        graph.toggle_edge(b, d);
      }
    }
  }

  void QuantumGraphState::myr_graph(uint32_t a, bool outcome) {
    uint32_t gate_id = outcome ? SQRTZDGATE : SQRTZGATE;
    std::vector<uint32_t> ngbh = graph.neighbors(a);

    for (auto n : ngbh) {
      apply_gater(n, gate_id);
    }
    apply_gater(a, gate_id);

    ngbh.push_back(a);

    for (auto i : ngbh) {
      for (auto j : ngbh) {
        if (i < j) {
          graph.toggle_edge(i, j);
        }
      }
    }
  }

  void QuantumGraphState::mzr_graph(uint32_t a, bool outcome) {
    std::vector<uint32_t> ngbh = graph.neighbors(a);

    for (auto n : ngbh) {
      graph.remove_edge(a, n);
      if (outcome) {
        apply_gater(n, ZGATE);
      }
    }

    if (outcome) {
      apply_gater(a, XGATE);
    }

    apply_gater(a, HGATE);
  }

  std::string QuantumGraphState::to_string() const {
    return graph.to_string();
  }

  void QuantumGraphState::x(uint32_t a) {
    apply_gatel(a, XGATE);
  }

  void QuantumGraphState::y(uint32_t a) {
    apply_gatel(a, YGATE);
  }

  void QuantumGraphState::z(uint32_t a) {
    apply_gatel(a, ZGATE);
  }

  void QuantumGraphState::h(uint32_t a) {
    apply_gatel(a, HGATE);
  }

  void QuantumGraphState::s(uint32_t a) {
    apply_gatel(a, SGATE);
  }

  void QuantumGraphState::sd(uint32_t a) {
    apply_gatel(a, SDGATE);
  }

  void QuantumGraphState::cz(uint32_t a, uint32_t b) {
    assert((a < num_qubits) && (b < num_qubits) && (a != b));

    if (!isolated(a, b))  {
      remove_vop(a, b);
    }
    if (!isolated(b, a)) { 
      remove_vop(b, a);
    }
    if (!isolated(a, b)) { 
      remove_vop(a, b);
    }

    uint32_t lookup[3];
    for (uint32_t i = 0; i < 3; i ++) {
      lookup[i] = CZ_LOOKUP[graph.get_val(a)][graph.get_val(b)][graph.contains_edge(a, b)][i];
    }

    graph.set_val(a, lookup[0]);
    graph.set_val(b, lookup[1]);

    if (lookup[2] != graph.contains_edge(a, b)) {
      graph.toggle_edge(a, b);
    }
  }

  double QuantumGraphState::mzr_expectation(uint32_t a) {
    uint32_t basis = CONJUGATION_TABLE[graph.get_val(a)];
    bool positive = basis > 3;
    if ((basis == 1) || (basis == 4)) {
      if (graph.degree(a) == 0) { 
        return 2*int(positive) - 1.0;
      }
    }

    return 0.0;
  }


  bool QuantumGraphState::mzr(uint32_t a) {
    uint32_t basis = CONJUGATION_TABLE[graph.get_val(a)];
    bool positive = basis <= 3;

    if ((basis == 1) || (basis == 4)) {
      if (graph.degree(a) == 0) {
        return !positive;
      }
    }

    bool outcome = rand() % 2;
    bool real_outcome = positive ? outcome : !outcome;

    if ((basis == 1) || (basis == 4)) {
      mxr_graph(a, real_outcome);
    } else if ((basis == 2) || (basis == 5))  {
      myr_graph(a, real_outcome);
    } else if ((basis == 3) || (basis == 6)) {
      mzr_graph(a, real_outcome);
    }

    return outcome;
  }

  void QuantumGraphState::toggle_edge_gate(uint32_t a, uint32_t b) {
    uint32_t ca = graph.get_val(a);
    uint32_t cb = graph.get_val(b);

    apply_gatel(a, HERMITIAN_CONJUGATE_TABLE[ca]);
    apply_gatel(b, HERMITIAN_CONJUGATE_TABLE[cb]);
    cz(a, b);
    apply_gatel(a, ca);
    apply_gatel(b, cb);
  }
  
  double QuantumGraphState::graph_state_entropy(const std::vector<uint32_t> &qubits, Graph<> &graph) {
    Graph<int, bool> bipartite_graph = graph.partition(qubits);
    int s = 2*bipartite_graph.num_vertices;
    for (uint32_t i = 0; i < bipartite_graph.num_vertices; i++) {
      if (bipartite_graph.get_val(i)) {
        s--;
      }
    }


    // Trim leafs in B
    bool found_isolated = true;
    while (!found_isolated) {
      found_isolated = false;
      for (uint32_t i = 0; i < bipartite_graph.num_vertices; i++) {
        if (bipartite_graph.degree(i) == 1 && !bipartite_graph.get_val(i)) {
          found_isolated = true;
          uint32_t neighbor = bipartite_graph.neighbors(i)[0];
          bipartite_graph.remove_vertex(std::max(i, neighbor));
          bipartite_graph.remove_vertex(std::min(i, neighbor));
          s -= 2;

          break;
        }
      }
    }

    bool continue_deleting = true;
    while (continue_deleting) {
      continue_deleting = false;

      uint32_t del_node;
      for (uint32_t i = 0; i < bipartite_graph.num_vertices; i++) {
        if (!bipartite_graph.get_val(i)) { // Vertex is in B; must delete it.
          continue_deleting = true;
          del_node = i;
          break;
        }
      }

      if (!continue_deleting) {
        break;
      }

      uint32_t del_node_degree = bipartite_graph.degree(del_node);
      if (del_node_degree == 0) {
        bipartite_graph.remove_vertex(del_node);
        s -= 2;
      } else if (del_node_degree == 1) {
        uint32_t neighbor = bipartite_graph.neighbors(del_node)[0];
        bipartite_graph.remove_vertex(std::max(del_node, neighbor));
        bipartite_graph.remove_vertex(std::min(del_node, neighbor));
        s -= 2;
      } else {
        bool found_pivot = false;
        uint32_t pivot = 0;
        uint32_t min_degree = INT_MAX;
        for (auto neighbor : bipartite_graph.neighbors(del_node)) {
          uint32_t deg = bipartite_graph.degree(neighbor);

          if ((deg != 1) && (deg < min_degree)) {
            found_pivot = true;
            min_degree = deg;
            pivot = neighbor;
          }
        }

        // there is no valid pivot, so subgraph is a simple tree; clear it.
        if (!found_pivot) {
          std::vector<uint32_t> neighbors = bipartite_graph.neighbors(del_node);
          neighbors.push_back(del_node);
          std::sort(neighbors.begin(), neighbors.end(), std::greater<>());

          for (auto neighbor : neighbors) {
            bipartite_graph.remove_vertex(neighbor);
          }

          s -= neighbors.size();
        } else {
          for (auto neighbor : bipartite_graph.neighbors(del_node)) {
            if (neighbor != pivot) {
              std::vector<uint32_t> pivot_neighbors = bipartite_graph.neighbors(pivot);
              for (uint32_t k = 0; k < min_degree; k++) {
                uint32_t pivot_neighbor = pivot_neighbors[k];
                bipartite_graph.toggle_edge(neighbor, pivot_neighbor);
              }
            }
          }

          // Pivot completed; ready to be deleted on the next iteration
        }

      }
    }

    for (uint32_t i = 0; i < bipartite_graph.num_vertices; i++) {
      if (bipartite_graph.degree(i) == 0) {
        s--;
      }
    }

    return static_cast<double>(s);
  }

  double QuantumGraphState::entropy(const std::vector<uint32_t> &qubits, uint32_t index) {
    return QuantumGraphState::graph_state_entropy(qubits, graph);
  }

  double QuantumGraphState::sparsity() const {
    double s = 0.;
    for (uint32_t i = 0; i < num_qubits; i++) {
      for (uint32_t j = 0; j < num_qubits; j++) {
        s += graph.adjacency_matrix(i, j);
      }
    }

    return s/(num_qubits*num_qubits);
  }
