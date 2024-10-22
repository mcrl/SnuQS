#ifndef _GATE_OPERATIONS_SLICED_H_
#define _GATE_OPERATIONS_SLICED_H_
#include "operation.h"

class IdentitySliced : public GateOperationSliced {
 public:
  IdentitySliced(std::vector<size_t> targets,
                 std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~IdentitySliced();
  virtual bool diagonal() const override;
  virtual bool anti_diagonal() const override;
  virtual bool sliceable() const override;
};
#endif  //_GATE_OPERATIONS_SLICED_H_
