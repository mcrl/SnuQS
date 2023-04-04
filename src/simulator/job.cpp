#include "job.h"

#include <string>
#include <chrono>

#include "simulator/simulator.h"
#include "logger.h"

namespace snuqs {

Job::Job(Simulator *sim)
: sim_(sim) {
}

Job::~Job() {
}

void Job::Run(const std::vector<QuantumCircuit> &circs) {
	sim_->init(circs[0].num_qubits());

	auto s = std::chrono::system_clock::now();

	sim_->run(circs);
	auto e = std::chrono::system_clock::now();

  std::chrono::duration<double> t = (e-s);
  Logger::info("Total Execution time: {}\n", t.count());

	sim_->deinit();
}

} // namespace snuqs
