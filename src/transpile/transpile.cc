#include "transpile/transpile.h"
#include "assertion.h"
#include "circuit/arg.h"
#include "circuit/qop.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <set>

namespace snuqs {

std::shared_ptr<Circuit> transpileSingleGPU(Circuit &_circ, size_t num_qubits) {
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(_circ.name());

  auto &qregs = _circ.qregs();
  for (auto qreg : _circ.qregs()) {
    circ->append_qreg(qreg);
  }

  for (auto creg : _circ.cregs()) {
    circ->append_creg(creg);
  }

  circ->append(std::make_shared<InitZeroState>());
  for (auto qop : _circ.qops()) {
    circ->append(qop);
  }

  auto &last_qreg = qregs[qregs.size() - 1];
  circ->append(std::make_shared<MemcpyD2H>());
  circ->append(std::make_shared<Sync>());

  return circ;
}

std::shared_ptr<Circuit> transpileMultiGPU(Circuit &_circ, int num_qubits,
                                           int device, int num_devices) {
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(_circ.name());
  int num_qubits_per_device = num_qubits - int(log2(num_devices));

  auto &qregs = _circ.qregs();
  for (auto qreg : qregs) {
    circ->append_qreg(qreg);
  }

  for (auto creg : _circ.cregs()) {
    circ->append_creg(creg);
  }

  if (device == 0) {
    circ->append(std::make_shared<InitZeroState>());
  } else {
    circ->append(std::make_shared<SetZero>());
  }

  auto &qops = _circ.qops();
  for (size_t i = 0; i < qops.size(); ++i) {
    auto qop = qops[i];

    std::vector<std::shared_ptr<Qarg>> non_local_args;
    std::set<size_t> local_indices;
    auto qargs = qop->qargs();
    for (auto &qarg : qargs) {
      size_t index = qarg->globalIndex();
      if (index >= num_qubits_per_device) {
        non_local_args.push_back(qarg);
      } else {
        local_indices.insert(index);
      }
    }
    //
    if (non_local_args.size() > 0) {
      std::vector<std::shared_ptr<Qarg>> target_args;
      for (int t = num_qubits_per_device - 1; t >= 0; --t) {
        if (local_indices.find(t) == local_indices.end()) {
          target_args.push_back(
              std::make_shared<Qarg>(_circ.getQregForIndex(t), t));
          if (target_args.size() == non_local_args.size())
            break;
        }
      }
      assert(target_args.size() == non_local_args.size());

      std::sort(non_local_args.begin(), non_local_args.end(),
                [](auto a, auto b) { return *b < *a; });

      std::map<std::shared_ptr<Qarg>, std::shared_ptr<Qarg>> qarg_map;

      for (size_t j = 0; j < non_local_args.size(); ++j) {
        qarg_map[non_local_args[j]] = target_args[j];
      }
      std::vector<std::shared_ptr<Qarg>> new_qargs;
      for (auto &qarg : qargs) {
        size_t index = qarg->globalIndex();
        if (index >= num_qubits_per_device) {
          new_qargs.push_back(qarg_map[qarg]);
        } else {
          new_qargs.push_back(qarg);
        }
      }

      auto new_qop = qop->clone();
      new_qop->setQargs(new_qargs);

      for (size_t j = 0; j < non_local_args.size(); ++j) {
        circ->append(
            std::make_shared<GlobalSwap>(std::vector<std::shared_ptr<Qarg>>{
                non_local_args[j], target_args[j]}));
      }

      circ->append(new_qop);

      for (size_t j = 0; j < non_local_args.size(); ++j) {
        circ->append(
            std::make_shared<GlobalSwap>(std::vector<std::shared_ptr<Qarg>>{
                non_local_args[j], target_args[j]}));
      }
    } else {
      circ->append(qop);
    }
  }

  auto &last_qreg = qregs[qregs.size() - 1];
  circ->append(std::make_shared<MemcpyD2H>());
  circ->append(std::make_shared<Sync>());

  return circ;
}

std::map<Qarg, Qarg>
qargMapping(Circuit &_circ, size_t num_qubits_per_device,
            std::vector<std::shared_ptr<Qarg>> &global_args,
            std::set<size_t> &local_indices) {
  std::vector<std::shared_ptr<Qarg>> target_args;
  for (int t = num_qubits_per_device - 1; t >= 0; --t) {
    if (local_indices.find(t) == local_indices.end()) {
      target_args.push_back(
          std::make_shared<Qarg>(_circ.getQregForIndex(t), t));
      if (target_args.size() == global_args.size())
        break;
    }
  }
  assert(target_args.size() == global_args.size());

  std::sort(global_args.begin(), global_args.end(),
            [](auto a, auto b) { return *b < *a; });

  std::map<Qarg, Qarg> qarg_map;

  for (size_t j = 0; j < global_args.size(); ++j) {
    qarg_map[*target_args[j]] = *global_args[j];
    qarg_map[*global_args[j]] = *target_args[j];
  }
  return qarg_map;
}

std::shared_ptr<Circuit> transpileCPU(Circuit &_circ, int num_qubits_per_device,
                                      int num_qubits, int device,
                                      int num_devices) {
  ///////////////////////////////////////
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(_circ.name());
  size_t num_qubits_per_slice = num_qubits_per_device + int(log2(num_devices));
  size_t num_states_per_slice = (1ull << num_qubits_per_slice);
  size_t num_slices = (1ull << (num_qubits - num_qubits_per_slice));

  auto &qregs = _circ.qregs();
  for (auto qreg : qregs) {
    circ->append_qreg(qreg);
  }
  auto &last_qreg = qregs[qregs.size() - 1];

  for (auto creg : _circ.cregs()) {
    circ->append_creg(creg);
  }

  auto &qops = _circ.qops();
  int prev_i = 0;
  std::map<Qarg, Qarg> qarg_map;

  for (int i = 0; i < qops.size(); ++i) {
    auto qop = qops[i];
    std::vector<std::shared_ptr<Qarg>> global_args;
    std::set<size_t> local_indices;
    auto qargs = qop->qargs();
    for (auto &qarg : qargs) {
      size_t index = qarg->globalIndex();
      if (index >= num_qubits_per_slice) {
        global_args.push_back(qarg);
      } else {
        local_indices.insert(index);
      }
    }

    if (global_args.size() > 0) {
      for (int s = 0; s < num_slices; ++s) {
        circ->append(std::make_shared<Slice>(s));
        if (prev_i == 0) {
          if (s == 0 && device == 0) {
            circ->append(std::make_shared<InitZeroState>());
          } else {
            circ->append(std::make_shared<SetZero>());
          }
        } else {
          circ->append(std::make_shared<MemcpyH2D>(qarg_map, s));
        }

        for (int j = prev_i; j < i; ++j) {
          circ->append(qops[j]);
        }

        circ->append(std::make_shared<MemcpyD2H>(qarg_map, s));
        if (s + 1 < num_slices) {
          circ->append(std::make_shared<MemcpyH2D>(qarg_map, s + 1));
        }
      }
      circ->append(std::make_shared<Sync>());
      qarg_map =
          qargMapping(_circ, num_qubits_per_device, global_args, local_indices);

      for (int j = i; j < qops.size(); ++j) {
        auto qop = qops[i];
        std::vector<std::shared_ptr<Qarg>> new_qargs;
        for (int k = 0; k < qop->qargs_.size(); ++k) {
          size_t index = qop->qargs_[k]->globalIndex();

          if (qarg_map.find(*qop->qargs_[k]) != qarg_map.end()) {
            qop->qargs_[k] = std::make_shared<Qarg>(qarg_map[*qop->qargs_[k]]);
          }
        }
      }
      prev_i = i;
    }
  }

  if (prev_i != qops.size()) {
  }

  return circ;
}

} // namespace snuqs
