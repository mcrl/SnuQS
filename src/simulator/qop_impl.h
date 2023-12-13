#ifndef __QOP_IMPL_H__
#define __QOP_IMPL_H__

#include "cuda_api.h"
#include <complex>
#include <vector>

namespace snuqs {
namespace cuda {

template <typename T> class QopImpl {
public:
  static void initZeroState(std::complex<T> *buffer, size_t count);
  static void setZero(std::complex<T> *buffer, size_t count);

  static void id(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void x(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void y(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void z(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void h(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void s(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void sdg(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void t(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void tdg(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void sx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void sxdg(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params);

  static void p(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void rx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void ry(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void rz(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void u0(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void u1(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void u2(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void u3(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void u(std::complex<T> *buffer, size_t count,
                std::vector<size_t> targets, std::vector<double> params);

  static void cx(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void cy(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void cz(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void swap(std::complex<T> *buffer, size_t count,
                   std::vector<size_t> targets, std::vector<double> params);

  static void ch(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void csx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void crx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void cry(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void crz(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void cp(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void cu1(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void rxx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void rzz(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void cu3(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void cu(std::complex<T> *buffer, size_t count,
                 std::vector<size_t> targets, std::vector<double> params);

  static void ccx(std::complex<T> *buffer, size_t count,
                  std::vector<size_t> targets, std::vector<double> params);

  static void cswap(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::vector<double> params);

  static void global_swap(std::complex<T> *buffer, size_t count,
                          std::vector<size_t> targets,
                          std::vector<double> params,
                          std::complex<T> *mem_buffer);
  static void flush(std::complex<T> *buffer, size_t count,
                    std::vector<size_t> targets, std::complex<T> *mem_buffer);
};

} // namespace cuda
} // namespace snuqs

#endif // __QOP_IMPL_H__
