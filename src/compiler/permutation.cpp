#include "permutation.h"

namespace snuqs {

Permutation::Permutation(const std::vector<size_t> &perm)
: perm_(perm) {
}

Permutation Permutation::inv() const {
	std::vector<size_t> inv_map(perm_.size());

	for (size_t i = 0; i < perm_.size(); ++i) {
		inv_map[perm_[i]] = i;
	}

	return Permutation(inv_map);
}

size_t Permutation::size() const {
	return perm_.size();
}

size_t Permutation::operator[](size_t i) const {
	return perm_[i];
}

size_t& Permutation::operator[](size_t i) {
	return perm_[i];
}

} // namespace snuqs
