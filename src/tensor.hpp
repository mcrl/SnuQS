#pragma once
#include <string>
#include <vector>
#include "configure.hpp"

namespace snuqs {

class Tensor {

	public:
	Tensor(std::vector<unsigned int> inedges, std::vector<unsigned int> outedges, std::vector<snuqs::amp_t> mem);

	void setInEdges(std::vector<unsigned int> inedges);
	void setOutEdges(std::vector<unsigned int> outedges);

	std::vector<unsigned int> inedges_;
	std::vector<unsigned int> outedges_;
	std::vector<snuqs::amp_t> mem_;

    friend std::ostream& operator<<(std::ostream &os, const Tensor &gate); 
    virtual std::ostream& operator<<(std::ostream &os) const;

    snuqs::amp_t *d_mem_;
};

} // namespace snuqs
