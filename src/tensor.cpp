#include "tensor.hpp"

namespace snuqs {

Tensor::Tensor(std::vector<unsigned int> inedges, std::vector<unsigned int> outedges, std::vector<snuqs::amp_t> mem) 
: inedges_(inedges)
, outedges_(outedges)
, mem_(mem)
{
}

void Tensor::setInEdges(std::vector<unsigned int> inedges)
{
	inedges_ = inedges;
}

void Tensor::setOutEdges(std::vector<unsigned int> outedges)
{
	outedges_ = outedges;
}

std::ostream& Tensor::operator<<(std::ostream &os) const 
{
	os << "{";

	for (size_t i = 0; i < inedges_.size(); ++i) {
		os << inedges_[i];
		if (i+1 < inedges_.size())
			os << ",";
	}
	os << "}";
	os << "->";
	os << "{";
	for (size_t i = 0; i < outedges_.size(); ++i) {
		os << outedges_[i];
		if (i+1 < outedges_.size())
			os << ",";
	}
	os << "}";

	return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) 
{
    return tensor.operator<<(os);
}

} // namespace snuqs
