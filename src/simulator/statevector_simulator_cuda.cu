#include <complex>
#include <cstdint>
#include <vector>

#include "assertion.h"

#include "cuda_api.h"
#include "statevector_simulator_cuda.h"

namespace snuqs {
namespace cuda {

template <typename T> struct complex {
  T real;
  T imag;

  __device__ complex(T real, T imag) {
    this->real = real;
    this->imag = imag;
  }

  __device__ complex<T> &operator=(double other) {
    this->real = other;
    this->imag = 0.;
    return *this;
  }

  __device__ const complex<T> &operator+(const double &other) const {
    T real = this->real + other;
    T imag = this->imag;
    return complex<T>(real, imag);
  }

  __device__ const complex<T> &operator-(const double &other) const {
    T real = this->real - other;
    T imag = this->imag;
    return complex<T>(real, imag);
  }

  __device__ const complex<T> &operator*(const double &other) const {
    T real = this->real * other;
    T imag = this->imag * other;
    return complex<T>(real, imag);
  }

  __device__ complex<T> &operator=(const complex &other) {
    this->real = other.real;
    this->imag = other.imag;
    return *this;
  }

  __device__ const complex<T> operator+(const complex &other) const {
    T real = this->real + other->real;
    T imag = this->imag + other->imag;
    return complex(real, imag);
  }

  __device__ const complex<T> operator-(const complex &other) const {
    T real = this->real - other->real;
    T imag = this->imag - other->imag;
    return complex(real, imag);
  }

  __device__ const complex<T> operator*(const complex &other) const {
    T real = this->real * other->real - this->imag * other->imag;
    T imag = this->real * other->imag + this->imag * other->real;
    return complex(real, imag);
  }
};

namespace kernel {

template <typename T>
__global__ void initZero(cuda::complex<T> *buffer, size_t count) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= count)
    return;

  buffer[i] = 0.;
  if (i == 0)
    buffer[0] = 1.;
}

template <typename T>
__global__ void oneQubitGate(cuda::complex<T> *buffer, size_t count,
                             const size_t target, std::vector<double> params) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t s = (1ul << target);
  size_t idx = ((i >> target) << (target + 1)) + (i & (s - 1));

  cuda::complex<T> a0 = buffer[idx];
  cuda::complex<T> a1 = buffer[idx + s];
  buffer[idx] = a0 + a1;
  buffer[idx + s] = a0 / a1;
}

template <typename T>
__global__ void controlledOneQubitGate(cuda::complex<T> *buffer, size_t count,
                                       const size_t control,
                                       const size_t target,
                                       std::vector<double> params) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t target_small = (control < target) ? control : target;
  size_t target_large = (control > target) ? control : target;
  size_t st0 = (1ul << target_small);
  size_t st1 = (1ul << target_large);
  size_t cst = 1ul << control;
  size_t st = st0 + st1;

  size_t ci = ((i >> target_small) << (target_small + 1)) + (i & (st0 - 1));
  size_t idx = ((ci >> target_large) << (target_large + 1)) + (ci & (st1 - 1));
}

} // namespace kernel

template <typename T>
void gate<T>::initZero(std::complex<T> *buffer, size_t count,
                       std::vector<size_t> targets,
                       std::vector<double> params) {
  kernel::initZero<T><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count);
  api::assertKernelLaunch();
}

template <typename T>
void gate<T>::id(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  /* Do Nothing */
}

template <typename T>
void gate<T>::x(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  //  kernel::oneQubitGate<T><<<(count + 255) / 256, 256>>>(
  //      reinterpret_cast<cuda::complex<T> *>(buffer), targets[0], params[0]);
  api::assertKernelLaunch();
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::y(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::z(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::h(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::s(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::sdg(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::t(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::tdg(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::sx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::sxdg(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::p(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::ry(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rz(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::u0(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::u1(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::u2(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::u3(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::u(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cy(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cz(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::swap(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::ch(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::csx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::crx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cry(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::crz(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cp(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cu1(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rxx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rzz(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cu3(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cu(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::ccx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::cswap(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rccx(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::rc3x(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::c3x(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::c3sqrtx(std::complex<T> *buffer, size_t count,
                      std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template <typename T>
void gate<T>::c4x(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params) {
  NOT_IMPLEMENTED();
}

template class gate<double>;
template class gate<float>;

} // namespace cuda
} // namespace snuqs
