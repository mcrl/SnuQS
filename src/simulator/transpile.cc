#include "simulator/transpile.h"
#include "assertion.h"
#include "circuit/arg.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <set>

namespace snuqs {

std::shared_ptr<Circuit> transpileSingleGPU(Circuit &_circ) {
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(_circ.name());

  for (auto qreg : _circ.qregs()) {
    circ->append_qreg(qreg);
  }

  for (auto creg : _circ.cregs()) {
    circ->append_creg(creg);
  }

  for (auto qop : _circ.qops()) {
    circ->append(qop);
  }

  return circ;
}

std::shared_ptr<Circuit> transpileMultiGPU(Circuit &_circ, int num_qubits,
                                           int num_devices) {
  std::shared_ptr<Circuit> circ = std::make_shared<Circuit>(_circ.name());
  int num_qubits_per_device = num_qubits - int(log2(num_devices));

  for (auto qreg : _circ.qregs()) {
    circ->append_qreg(qreg);
  }

  for (auto creg : _circ.cregs()) {
    circ->append_creg(creg);
  }

  auto &qops = _circ.qops();
  for (size_t i = 0; i < qops.size(); ++i) {
    auto qop = qops[i];

    std::vector<std::shared_ptr<Qarg>> global_args;
    std::set<size_t> local_indices;
    auto qargs = qop->qargs();
    for (auto &qarg : qargs) {
      size_t index = qarg->globalIndex();
      if (index >= num_qubits_per_device) {
        global_args.push_back(qarg);
      } else {
        local_indices.insert(index);
      }
    }
    // FIXME: What if multiple qregs?
    //
    if (global_args.size() > 0) {
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

      std::map<std::shared_ptr<Qarg>, std::shared_ptr<Qarg>> qarg_map;

      for (size_t j = 0; j < global_args.size(); ++j) {
        qarg_map[global_args[j]] = target_args[j];
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

      qop->setQargs(new_qargs);

      for (size_t j = 0; j < global_args.size(); ++j) {
        circ->append(
            std::make_shared<GlobalSwap>(std::vector<std::shared_ptr<Qarg>>{
                global_args[j], target_args[j]}));
      }

      circ->append(qop);

      for (size_t j = 0; j < global_args.size(); ++j) {
        circ->append(
            std::make_shared<GlobalSwap>(std::vector<std::shared_ptr<Qarg>>{
                global_args[j], target_args[j]}));
      }
    } else {
      circ->append(qop);
    }
  }

  return circ;
}

} // namespace snuqs
