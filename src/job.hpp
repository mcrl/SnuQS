#pragma once

#include <string>
#include <memory>
#include <chrono>
#include "job.hpp"
#include "quantumCircuit.hpp"
#include "gpu_utils.hpp"

#include "simulator.hpp"

namespace snuqs {


class Job {
	private:
	const std::vector<snuqs::QuantumCircuit> &circs_;
	snuqs::Simulator::shared_ptr sim_;

	template<typename T>
	void printTime(std::string msg, T &s, T &e) {
		std::chrono::duration<double> t = (e-s);
		std::cout <<  msg << ": " << t.count() << "s\n";
	}

	public:
	using shared_ptr = std::shared_ptr<Job>;

	Job(const std::vector<snuqs::QuantumCircuit> &circ, snuqs::Simulator::shared_ptr sim);
	~Job();
	void run(); 
};

} // namespace snuqs
