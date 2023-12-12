#include "simulator/executor.h"
#include "assertion.h"
#include "simulator/qop_impl.h"
#include <vector>

#include <iostream>
namespace snuqs {
namespace cuda {

std::vector<size_t>
qargsToIndices(const std::vector<std::shared_ptr<Qarg>> &args) {
  std::vector<size_t> indices(args.size());
  for (int i = 0; i < args.size(); ++i) {
    indices[i] = args[i]->globalIndex();
  }
  return indices;
}

std::vector<double>
paramsToValues(const std::vector<std::shared_ptr<Parameter>> &params) {
  std::vector<double> values(params.size());

  for (int i = 0; i < params.size(); ++i) {
    values[i] = params[i]->eval();
  }
  return values;
}

template <typename T>
static void exec_gate(Qgate *qop, Buffer<T> *buffer, size_t num_states) {
  switch (qop->gate_type()) {
  case QgateType::ID:
    QopImpl<T>::id(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::X:
    QopImpl<T>::x(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::Y:
    QopImpl<T>::y(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::Z:
    QopImpl<T>::z(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::H:
    QopImpl<T>::h(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::S:
    QopImpl<T>::s(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::SDG:
    QopImpl<T>::sdg(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::T:
    QopImpl<T>::t(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::TDG:
    QopImpl<T>::tdg(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::SX:
    QopImpl<T>::sx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::SXDG:
    QopImpl<T>::sxdg(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                     paramsToValues(qop->params_));
    break;
  case QgateType::P:
    QopImpl<T>::p(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::RX:
    QopImpl<T>::rx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::RY:
    QopImpl<T>::ry(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::RZ:
    QopImpl<T>::rz(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::U0:
    QopImpl<T>::u0(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::U1:
    QopImpl<T>::u1(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::U2:
    QopImpl<T>::u2(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::U3:
    QopImpl<T>::u3(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::U:
    QopImpl<T>::u(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                  paramsToValues(qop->params_));
    break;
  case QgateType::CX:
    QopImpl<T>::cx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::CY:
    QopImpl<T>::cy(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::CZ:
    QopImpl<T>::cz(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::SWAP:
    QopImpl<T>::swap(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                     paramsToValues(qop->params_));
    break;
  case QgateType::CH:
    QopImpl<T>::ch(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::CSX:
    QopImpl<T>::csx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CRX:
    QopImpl<T>::crx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CRY:
    QopImpl<T>::cry(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CRZ:
    QopImpl<T>::crz(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CP:
    QopImpl<T>::cp(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::CU1:
    QopImpl<T>::cu1(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::RXX:
    QopImpl<T>::rxx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::RZZ:
    QopImpl<T>::rzz(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CU3:
    QopImpl<T>::cu3(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CU:
    QopImpl<T>::cu(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                   paramsToValues(qop->params_));
    break;
  case QgateType::CCX:
    QopImpl<T>::ccx(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                    paramsToValues(qop->params_));
    break;
  case QgateType::CSWAP:
    QopImpl<T>::cswap(buffer->ptr(), num_states, qargsToIndices(qop->qargs_),
                      paramsToValues(qop->params_));
    break;
  }
}

template <typename T>
void exec(Qop *qop, Buffer<T> *buffer, size_t num_states,
          Buffer<T> *mem_buffer) {
  switch (qop->type()) {
  case QopType::BARRIER:
    NOT_IMPLEMENTED();
  case QopType::RESET:
    NOT_IMPLEMENTED();
  case QopType::MEASURE:
    NOT_IMPLEMENTED();
  case QopType::COND:
    NOT_IMPLEMENTED();
  case QopType::CUSTOM:
    for (auto &qop : dynamic_cast<Custom *>(qop)->qops()) {
      exec<T>(qop.get(), buffer, num_states, mem_buffer);
    }
    break;
  case QopType::QGATE:
    exec_gate<T>(dynamic_cast<Qgate *>(qop), buffer, num_states);
    break;
  case QopType::GLOBAL_SWAP:
    QopImpl<T>::global_swap(buffer->ptr(), num_states,
                            qargsToIndices(qop->qargs_),
                            paramsToValues(qop->params_), mem_buffer->ptr());
  }
}

template void exec<float>(Qop *qop, Buffer<float> *buffer, size_t num_states,
                          Buffer<float> *mem_buffer);
template void exec<double>(Qop *qop, Buffer<double> *buffer, size_t num_states,
                           Buffer<double> *mem_buffer);

} // namespace cuda
} // namespace snuqs
