#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <complex>
#include <cstddef>
#include <vector>

class StateVector {
public:
  StateVector(size_t num_qubits);
  ~StateVector();
  std::complex<double> *data();
  size_t dim() const;
  size_t num_elems() const;
  std::vector<size_t> shape() const;
  std::string __repr__() const;

private:
  size_t num_qubits_;
  std::complex<double> *data_;
};

#endif //_STATE_VECTOR_H_
