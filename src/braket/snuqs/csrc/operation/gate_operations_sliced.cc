#include "gate_operations_sliced.h"

IdentitySliced::IdentitySliced(std::vector<size_t> targets,
                               std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperationSliced("IdentitySliced", targets, {}, ctrl_modifiers,
                          power) {
  ptr_[0] = 0;
}

IdentitySliced::~IdentitySliced() {}
bool IdentitySliced::diagonal() const { return true; }
bool IdentitySliced::anti_diagonal() const { return true; }
bool IdentitySliced::sliceable() const { return false; }
