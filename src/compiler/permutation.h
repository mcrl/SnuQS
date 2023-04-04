#pragma once

#include <vector>
#include <cstddef>

namespace snuqs {

class Permutation {
	public:
	Permutation(const std::vector<size_t> &p);

	size_t operator[](size_t i) const;
	size_t& operator[](size_t i);

	size_t size() const;

	Permutation inv() const;

	std::vector<size_t> perm_;
};

} // namespace snuqs
