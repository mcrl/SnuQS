#include "state_vector.h"

StateVector::StateVector(size_t num_qubits) : num_qubits_(num_qubits) {
  data_ = new std::complex<double>[1 << num_qubits];
}
StateVector::~StateVector() { delete[] data_; }
std::complex<double> *StateVector::data() { return data_; }
size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_qubits_); }
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
