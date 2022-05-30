#include "job.hpp"

namespace snuqs {

Job::Job(const std::vector<snuqs::QuantumCircuit> &circs, snuqs::Simulator::shared_ptr sim)
:
	circs_(circs),
	sim_(sim)
{
}

Job::~Job() 
{
}

void Job::run() {
	sim_->init(circs_[0].num_qubits());

	auto s = std::chrono::system_clock::now();

	sim_->run(circs_);
	auto e = std::chrono::system_clock::now();
	printTime("Total Execution Time", s, e);

	sim_->deinit();
}

} // namespace snuqs
