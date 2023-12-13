#include <algorithm>
#include <bitset>
#include <complex>
#include <cstdint>
#include <vector>

#include "assertion.h"

#include "cuda_api.h"
#include "qop_impl.h"

#include <iostream>
#include <random>

namespace snuqs {
namespace cuda {

template <typename T> struct complex {
  T real;
  T imag;

  __device__ complex(T real, T imag) {
    this->real = real;
    this->imag = imag;
  }

  __device__ complex<T> &operator+() { return *this; }

  __device__ complex<T> &operator-() {
    this->real = -this->real;
    this->imag = -this->imag;
    return *this;
  }

  __device__ complex<T> &operator=(T other) {
    this->real = other;
    this->imag = 0.;
    return *this;
  }

  __device__ const complex<T> operator+(const T &other) const {
    T real = this->real + other;
    T imag = this->imag;
    return complex<T>(real, imag);
  }

  __device__ const complex<T> operator-(const T &other) const {
    T real = this->real - other;
    T imag = this->imag;
    return complex<T>(real, imag);
  }

  __device__ const complex<T> operator*(const T &other) const {
    T real = this->real * other;
    T imag = this->imag * other;
    return complex<T>(real, imag);
  }

  __device__ complex<T> &operator=(const complex<T> &other) {
    this->real = other.real;
    this->imag = other.imag;
    return *this;
  }

  __device__ const complex<T> operator+(const complex<T> &other) const {
    T real = this->real + other.real;
    T imag = this->imag + other.imag;
    return complex(real, imag);
  }

  __device__ const complex<T> operator-(const complex<T> &other) const {
    T real = this->real - other.real;
    T imag = this->imag - other.imag;
    return complex(real, imag);
  }

  __device__ const complex<T> operator*(const complex<T> &other) const {
    T real = this->real * other.real - this->imag * other.imag;
    T imag = this->real * other.imag + this->imag * other.real;
    return complex(real, imag);
  }

  __device__ double abssq() const {
    return this->real * this->real + this->imag * this->imag;
  }
  __device__ static cuda::complex<T> expi(double angle) {
    return cuda::complex<T>(cos(angle), sin(angle));
  }
};

namespace kernel {

template <typename T>
__global__ void reset(cuda::complex<T> *buffer, size_t count, size_t target) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st = (1ul << target);

  if ((i & st) == 0) {

    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st];

    double prob = a0.abssq() + a1.abssq();
    buffer[i] = sqrt(prob);
    buffer[i + st] = 0;
  }
}

template <typename T>
__global__ void measure(cuda::complex<T> *buffer, size_t count, size_t target,
                        double rand) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st = (1ul << target);

  if ((i & st) == 0) {

    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st];

    if (a0.abssq() < rand) {
      buffer[i] = f0(a0, a1);
      buffer[i + st] = f1(a0, a1);
    }
  }
}

template <typename T>
__global__ void initZeroState(cuda::complex<T> *buffer, size_t count) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= count)
    return;

  buffer[i] = 0.;
  if (i == 0)
    buffer[0] = 1.;
}

template <typename T, typename F0, typename F1>
__global__ void oneQubitGate(cuda::complex<T> *buffer, size_t count,
                             const size_t target, F0 f0, F1 f1) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st = (1ul << target);

  if ((i & st) == 0) {

    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st];

    buffer[i] = f0(a0, a1);
    buffer[i + st] = f1(a0, a1);
  }
}

template <typename T, typename F0, typename F1, typename F2, typename F3>
__global__ void twoQubitGate(cuda::complex<T> *buffer, size_t count,
                             size_t target0, size_t target1, F0 f0, F1 f1,
                             F2 f2, F3 f3) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st0 = (1ul << target0);
  size_t st1 = (1ul << target1);

  if ((i & (st0 | st1)) == 0) {
    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st0];
    cuda::complex<T> a2 = buffer[i + st1];
    cuda::complex<T> a3 = buffer[i + st1 + st0];

    buffer[i] = f0(a0, a1, a2, a3);
    buffer[i + st0] = f1(a0, a1, a2, a3);
    buffer[i + st1] = f2(a0, a1, a2, a3);
    buffer[i + st1 + st0] = f3(a0, a1, a2, a3);
  }
}

template <typename T, typename F0, typename F1, typename F2, typename F3,
          typename F4, typename F5, typename F6, typename F7>
__global__ void threeQubitGate(cuda::complex<T> *buffer, size_t count,
                               size_t target0, size_t target1, size_t target2,
                               F0 f0, F1 f1, F2 f2, F3 f3, F4 f4, F5 f5, F6 f6,
                               F7 f7) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st0 = (1ul << target0);
  size_t st1 = (1ul << target1);
  size_t st2 = (1ul << target2);

  if ((i & (st0 | st1 | st2)) == 0) {
    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st0];
    cuda::complex<T> a2 = buffer[i + st1];
    cuda::complex<T> a3 = buffer[i + st1 + st0];
    cuda::complex<T> a4 = buffer[i + st2];
    cuda::complex<T> a5 = buffer[i + st2 + st0];
    cuda::complex<T> a6 = buffer[i + st2 + st1];
    cuda::complex<T> a7 = buffer[i + st2 + st1 + st0];

    buffer[i] = f0(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st0] = f1(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st1] = f2(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st1 + st0] = f3(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st2] = f4(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st2 + st0] = f5(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st2 + st1] = f6(a0, a1, a2, a3, a4, a5, a6, a7);
    buffer[i + st2 + st1 + st0] = f7(a0, a1, a2, a3, a4, a5, a6, a7);
  }
}

template <typename T, typename F0, typename F1, typename F2, typename F3,
          typename F4, typename F5, typename F6, typename F7, typename F8,
          typename F9, typename F10, typename F11, typename F12, typename F13,
          typename F14, typename F15>
__global__ void FourQubitGate(cuda::complex<T> *buffer, size_t count,
                              size_t target0, size_t target1, size_t target2,
                              size_t target3, F0 f0, F1 f1, F2 f2, F3 f3, F4 f4,
                              F5 f5, F6 f6, F7 f7, F8 f8, F9 f9, F10 f10,
                              F11 f11, F12 f12, F13 f13, F14 f14, F15 f15) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st0 = (1ul << target0);
  size_t st1 = (1ul << target1);
  size_t st2 = (1ul << target2);
  size_t st3 = (1ul << target3);

  if ((i & (st0 | st1 | st2 | st3)) == 0) {
    cuda::complex<T> a0 = buffer[i];
    cuda::complex<T> a1 = buffer[i + st0];
    cuda::complex<T> a2 = buffer[i + st1];
    cuda::complex<T> a3 = buffer[i + st1 + st0];
    cuda::complex<T> a4 = buffer[i + st2];
    cuda::complex<T> a5 = buffer[i + st2 + st0];
    cuda::complex<T> a6 = buffer[i + st2 + st1];
    cuda::complex<T> a7 = buffer[i + st2 + st1 + st0];
    cuda::complex<T> a8 = buffer[i + st3];
    cuda::complex<T> a9 = buffer[i + st3 + st0];
    cuda::complex<T> a10 = buffer[i + st3 + st1];
    cuda::complex<T> a11 = buffer[i + st3 + st1 + st0];
    cuda::complex<T> a12 = buffer[i + st3 + st2];
    cuda::complex<T> a13 = buffer[i + st3 + st2 + st0];
    cuda::complex<T> a14 = buffer[i + st3 + st2 + st1];
    cuda::complex<T> a15 = buffer[i + st3 + st2 + st1 + st0];
    buffer[i] = f0(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
                   a14, a15);
    buffer[i + st0] = f1(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                         a13, a14, a15);
    buffer[i + st1] = f2(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                         a13, a14, a15);
    buffer[i + st1 + st0] = f3(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                               a12, a13, a14, a15);
    buffer[i + st2] = f4(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                         a13, a14, a15);
    buffer[i + st2 + st0] = f5(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                               a12, a13, a14, a15);
    buffer[i + st2 + st1] = f6(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                               a12, a13, a14, a15);
    buffer[i + st2 + st1 + st0] = f7(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                                     a10, a11, a12, a13, a14, a15);
    buffer[i + st3] = f8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                         a13, a14, a15);
    buffer[i + st3 + st0] = f9(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                               a12, a13, a14, a15);
    buffer[i + st3 + st1] = f10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                a11, a12, a13, a14, a15);
    buffer[i + st3 + st1 + st0] = f11(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                                      a10, a11, a12, a13, a14, a15);
    buffer[i + st3 + st2] = f12(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                a11, a12, a13, a14, a15);
    buffer[i + st3 + st2 + st0] = f13(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                                      a10, a11, a12, a13, a14, a15);
    buffer[i + st3 + st2 + st1] = f14(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                                      a10, a11, a12, a13, a14, a15);
    buffer[i + st3 + st2 + st1 + st0] = f15(a0, a1, a2, a3, a4, a5, a6, a7, a8,
                                            a9, a10, a11, a12, a13, a14, a15);
  }
}

} // namespace kernel

template <typename T>
void QopImpl<T>::reset(std::complex<T> *buffer, size_t count,
                       std::vector<size_t> targets) {
  kernel::reset<T><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0]);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::initZeroState(std::complex<T> *buffer, size_t count) {
  kernel::initZeroState<T><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::setZero(std::complex<T> *buffer, size_t count) {
  api::memset(buffer, 0, sizeof(std::complex<T>) * count);
}

template <typename T>
void QopImpl<T>::id(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  /* Do Nothing */
}

template <typename T>
void QopImpl<T>::x(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a1;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::y(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a1 * cuda::complex<T>(0, -1);
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0 * cuda::complex<T>(0, 1);
  };
  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::z(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return -a0;
  };
  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::h(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {

  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return (a0 + a1) * M_SQRT1_2;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return (a0 - a1) * M_SQRT1_2;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::s(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a1 * cuda::complex<T>(0, 1);
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::sdg(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a1 * cuda::complex<T>(0, -1);
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::t(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(M_PI / 4);
    return a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::tdg(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(-M_PI / 4);
    return a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::sx(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0(1, 1);
    cuda::complex<T> coef1(1, -1);
    return (a0 * coef0 + a1 * coef1) * 0.5;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0(1, -1);
    cuda::complex<T> coef1(1, 1);
    return (a0 * coef0 + a1 * coef1) * 0.5;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::sxdg(std::complex<T> *buffer, size_t count,
                      std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0(1, -1);
    cuda::complex<T> coef1(1, 1);
    return (a0 * coef0 + a1 * coef1) * 0.5;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0(1, 1);
    cuda::complex<T> coef1(1, -1);
    return (a0 * coef0 + a1 * coef1) * 0.5;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::p(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  u1(buffer, count, targets, params);
}

template <typename T>
void QopImpl<T>::rx(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    double coef0 = cos(theta / 2);
    cuda::complex<T> coef1 = cuda::complex<T>(0, -1) * sin(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0 = cuda::complex<T>(0, -1) * sin(theta / 2);
    double coef1 = cos(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::ry(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    double coef0 = cos(theta / 2);
    double coef1 = -sin(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    double coef0 = sin(theta / 2);
    double coef1 = cos(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::rz(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double lambda = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0 = cuda::complex<T>::expi(-lambda / 2);
    return a0 * coef0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(lambda / 2);
    return a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::u0(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  /* Do Nothing */
}

template <typename T>
void QopImpl<T>::u1(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double lambda = params[0];

  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    return a0;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(lambda);
    return a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::u2(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double phi = params[0];
  double lambda = params[1];

  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    double coef0 = M_SQRT1_2;
    cuda::complex<T> coef1 = -cuda::complex<T>::expi(lambda) * M_SQRT1_2;
    return a0 * coef0 + a1 * coef1;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0 = cuda::complex<T>::expi(phi) * M_SQRT1_2;
    cuda::complex<T> coef1 = cuda::complex<T>::expi(phi + lambda) * M_SQRT1_2;
    return a0 * coef0 + a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::u3(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  u(buffer, count, targets, params);
}

template <typename T>
void QopImpl<T>::u(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];

  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    double coef0 = cos(theta / 2);
    cuda::complex<T> coef1 = -cuda::complex<T>::expi(lambda) * sin(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1) {
    cuda::complex<T> coef0 = cuda::complex<T>::expi(phi) * sin(theta / 2);
    cuda::complex<T> coef1 =
        cuda::complex<T>::expi(phi + lambda) * cos(theta / 2);
    return a0 * coef0 + a1 * coef1;
  };

  kernel::oneQubitGate<T, decltype(f0), decltype(f1)>
      <<<(count + 255) / 256, 256>>>(
          reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0], f0,
          f1);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cx(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a3; };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a1; };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cy(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    return a3 * cuda::complex<T>(0, -1);
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    return a1 * cuda::complex<T>(0, 1);
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cz(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a1; };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return -a3; };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::swap(std::complex<T> *buffer, size_t count,
                      std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a1; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a3; };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::ch(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    return (a1 + a3) * M_SQRT1_2;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    return (a1 - a3) * M_SQRT1_2;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::csx(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1(1, 1);
    cuda::complex<T> coef3(1, -1);
    return (a1 * coef1 + a3 * coef3) * 0.5;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1(1, -1);
    cuda::complex<T> coef3(1, 1);
    return (a1 * coef1 + a3 * coef3) * 0.5;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::crx(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef1 = cos(theta / 2);
    cuda::complex<T> coef3 = cuda::complex<T>(0, -1) * sin(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>(0, -1) * sin(theta / 2);
    double coef3 = cos(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cry(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef1 = cos(theta / 2);
    double coef3 = -sin(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef1 = sin(theta / 2);
    double coef3 = cos(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::crz(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double lambda = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(-lambda / 2);
    return a1 * coef1;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef3 = cuda::complex<T>::expi(lambda / 2);
    return a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cp(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  cu1(buffer, count, targets, params);
}

template <typename T>
void QopImpl<T>::cu1(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double lambda = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a1; };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef3 = cuda::complex<T>::expi(lambda);
    return a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::rxx(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef0 = cos(theta / 2);
    cuda::complex<T> coef3 = cuda::complex<T>(0, -1) * sin(theta / 2);
    return a0 * coef0 + a3 * coef3;
  };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef1 = cos(theta / 2);
    cuda::complex<T> coef2 = cuda::complex<T>(0, -1) * sin(theta / 2);
    return a1 * coef1 + a2 * coef2;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>(0, -1) * sin(theta / 2);
    double coef2 = cos(theta / 2);
    return a1 * coef1 + a2 * coef2;
  };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef0 = cuda::complex<T>(0, -1) * sin(theta / 2);
    double coef3 = cos(theta / 2);
    return a0 * coef0 + a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::rzz(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef0 = cuda::complex<T>::expi(-theta / 2);
    return a0 * coef0;
  };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(theta / 2);
    return a1 * coef1;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef2 = cuda::complex<T>::expi(theta / 2);
    return a2 * coef2;
  };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef3 = cuda::complex<T>::expi(-theta / 2);
    return a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cu3(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    double coef1 = cos(theta / 2);
    cuda::complex<T> coef3 = -cuda::complex<T>::expi(lambda) * sin(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(phi) * sin(theta / 2);
    cuda::complex<T> coef3 =
        cuda::complex<T>::expi(phi + lambda) * cos(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cu(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params) {
  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];
  double gamma = params[3];
  auto f0 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a0; };
  auto f1 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 = cuda::complex<T>::expi(gamma) * cos(theta / 2);
    cuda::complex<T> coef3 =
        -cuda::complex<T>::expi(gamma + lambda) * sin(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  auto f2 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2,
                           cuda::complex<T> a3) { return a2; };
  auto f3 = [=] __device__(cuda::complex<T> a0, cuda::complex<T> a1,
                           cuda::complex<T> a2, cuda::complex<T> a3) {
    cuda::complex<T> coef1 =
        cuda::complex<T>::expi(gamma + phi) * sin(theta / 2);
    cuda::complex<T> coef3 =
        cuda::complex<T>::expi(gamma + phi + lambda) * cos(theta / 2);
    return a1 * coef1 + a3 * coef3;
  };
  kernel::twoQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], f0, f1, f2, f3);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::ccx(std::complex<T> *buffer, size_t count,
                     std::vector<size_t> targets, std::vector<double> params) {
  auto f0 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a0; };
  auto f1 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a1; };
  auto f2 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a2; };
  auto f3 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a7; };
  auto f4 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a4; };
  auto f5 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a5; };
  auto f6 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a6; };
  auto f7 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a3; };

  kernel::threeQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], targets[2], f0, f1, f2, f3, f4, f5, f6, f7);
  api::assertKernelLaunch();
}

template <typename T>
void QopImpl<T>::cswap(std::complex<T> *buffer, size_t count,
                       std::vector<size_t> targets,
                       std::vector<double> params) {
  auto f0 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a0; };
  auto f1 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a1; };
  auto f2 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a2; };
  auto f3 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a5; };
  auto f4 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a4; };
  auto f5 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a3; };
  auto f6 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a6; };
  auto f7 = [=] __device__(
                cuda::complex<T> a0, cuda::complex<T> a1, cuda::complex<T> a2,
                cuda::complex<T> a3, cuda::complex<T> a4, cuda::complex<T> a5,
                cuda::complex<T> a6, cuda::complex<T> a7) { return a7; };

  kernel::threeQubitGate<T, decltype(f0)><<<(count + 255) / 256, 256>>>(
      reinterpret_cast<cuda::complex<T> *>(buffer), count, targets[0],
      targets[1], targets[2], f0, f1, f2, f3, f4, f5, f6, f7);
  api::assertKernelLaunch();
}

static size_t transform_block_num(size_t block_num, size_t q0, size_t q1) {
  // q0 < q1
  size_t diff = q1 - q0;
  std::bitset<64> bitset_tmp{block_num};
  std::bitset<64> bitset{block_num};
  bitset[0] = bitset_tmp[diff];
  bitset[diff] = bitset_tmp[0];
  return bitset.to_ulong();
}

static size_t transform_block_num(size_t block_num, size_t min_target,
                                  std::map<size_t, size_t> index_map) {
  if (min_target >= 64)
    return block_num;
  // q0 < q1

  std::bitset<64> bitset_tmp{block_num};
  std::bitset<64> bitset{block_num};
  for (auto &kv : index_map) {
    bitset[kv.first - min_target] = bitset_tmp[kv.second - min_target];
  }
  return bitset.to_ulong();
}

template <typename T>
void QopImpl<T>::global_swap(std::complex<T> *buffer, size_t count,
                             std::vector<size_t> targets,
                             std::vector<double> params,
                             std::complex<T> *mem_buffer) {
  size_t target0 = std::min(targets[0], targets[1]);
  size_t target1 = std::max(targets[0], targets[1]);

  int device, num_devices;
  api::getDevice(&device);
  api::getDeviceCount(&num_devices);

  size_t num_states_per_device = count;
  size_t num_states_per_block = (1ull << target0);
  size_t num_blocks_per_device = num_states_per_device / num_states_per_block;

  for (size_t i = 0; i < num_blocks_per_device; ++i) {
    size_t block_num = device * num_blocks_per_device + i;
    size_t changed_num = transform_block_num(block_num, target0, target1);

    api::memcpyAsync(&mem_buffer[changed_num * num_states_per_block],
                     &buffer[i * num_states_per_block],
                     num_states_per_block * sizeof(std::complex<T>),
                     cudaMemcpyDeviceToHost, 0);
  }

  api::deviceSynchronize();
#pragma omp barrier

  for (size_t i = 0; i < num_blocks_per_device; ++i) {
    size_t block_num = device * num_blocks_per_device + i;
    size_t changed_num = transform_block_num(block_num, target0, target1);
    api::memcpyAsync(&buffer[i * num_states_per_block],
                     &mem_buffer[changed_num * num_states_per_block],
                     num_states_per_block * sizeof(std::complex<T>),
                     cudaMemcpyHostToDevice, 0);
  }
}

template <typename T>
void QopImpl<T>::memcpy_h2d(std::complex<T> *buffer, size_t count,
                            std::map<size_t, size_t> index_map,
                            std::complex<T> *mem_buffer, size_t slice) {
  NOT_IMPLEMENTED();
}

template <typename T>
void QopImpl<T>::memcpy_d2h(std::complex<T> *buffer, size_t count,
                            std::map<size_t, size_t> index_map,
                            std::complex<T> *mem_buffer, size_t slice) {
  size_t min_target = -1;

  for (auto &kv : index_map) {
    min_target = std::min(min_target, kv.first);
  }

  int device, num_devices;
  api::getDevice(&device);
  api::getDeviceCount(&num_devices);

  size_t num_states_per_device = count;
  size_t num_states_per_block = (1ull << min_target);
  size_t num_blocks_per_device = num_states_per_device / num_states_per_block;

  for (size_t i = 0; i < num_blocks_per_device; ++i) {
    size_t block_num = slice * num_devices * num_blocks_per_device +
                       device * num_blocks_per_device + i;
    size_t changed_num = transform_block_num(block_num, min_target, index_map);

    api::memcpyAsync(&mem_buffer[changed_num * num_states_per_block],
                     &buffer[i * num_states_per_block],
                     num_states_per_block * sizeof(std::complex<T>),
                     cudaMemcpyDeviceToHost, 0);
  }
}

template <typename T>
void QopImpl<T>::sync(std::complex<T> *buffer, size_t count,
                      std::complex<T> *mem_buffer) {
  api::deviceSynchronize();
#pragma omp barrier
}

template class QopImpl<double>;
template class QopImpl<float>;

} // namespace cuda
} // namespace snuqs
