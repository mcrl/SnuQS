#include <boost/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "optionparser.h"
#include "job.hpp"
#include "quantumCircuit.hpp"
#include "simulator.hpp"
#include "snuqlCompiler.hpp"
#include "snuqlTranspiler.hpp"
#include "snuqlOptimizer.hpp"

#include "circuitAnalyzer.hpp"
#include "model.hpp"


#include "socl.hpp"

enum  optionIndex { UNKNOWN, HELP, SNUQL, METHOD, DEVICE, STORAGE, VERBOSE, MODEL, USEIO };

struct Arg: public option::Arg
{
	static void printError(const char* msg1, const option::Option& opt, const char* msg2)
	{
		fprintf(stderr, "%s", msg1);
		fwrite(opt.name, opt.namelen, 1, stderr);
		fprintf(stderr, "%s", msg2);
	}

	static option::ArgStatus Unknown(const option::Option& option, bool msg)
	{
		if (msg) printError("Unknown option '", option, "'\n");
		return option::ARG_ILLEGAL;
	}

	static option::ArgStatus Required(const option::Option& option, bool msg)
	{
		if (option.arg != 0)
			return option::ARG_OK;

		if (msg) printError("Option '", option, "' requires an argument\n");
		return option::ARG_ILLEGAL;
	}

	static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
	{
		if (option.arg != 0 && option.arg[0] != 0)
			return option::ARG_OK;

		if (msg) printError("Option '", option, "' requires a non-empty argument\n");
		return option::ARG_ILLEGAL;
	}

	static option::ArgStatus Numeric(const option::Option& option, bool msg)
	{
		char* endptr = 0;
		if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
		if (endptr != option.arg && *endptr == 0)
			return option::ARG_OK;

		if (msg) printError("Option '", option, "' requires a numeric argument\n");
		return option::ARG_ILLEGAL;
	}
};

const option::Descriptor usage[] =
{
	{UNKNOWN, 0, "", "",option::Arg::None,
		"USAGE: example [options]\n\n"
			"Options:" },
	{HELP, 0,"", "help", option::Arg::None,
		"  --help  \tPrint usage and exit." },
	{SNUQL, 0,"s","snuql", Arg::Required,
		"  --snuql, -s  \tsnuql file name."},
	{METHOD, 0,"m","method", Arg::Optional,
		"  --method, -m  \tsimulation method. default: statevector."},
	{DEVICE, 0,"d","device", Arg::Optional,
		"  --device, -d  \tsimulation device. default: cpu."},
	{STORAGE, 0,"s","storage", Arg::Optional,
		"  --storage, -s  \tstorage description file name."},
	{VERBOSE, 0,"v","verbose", Arg::Optional,
		"  --verbose, -v  \tanalyze quantum circuit."},
	{USEIO, 0,"i","io", Arg::Optional,
		"  --io, -i  \tanalyze quantum circuit."},
	{MODEL, 0,"m","model", Arg::Optional,
		"  --model, -m  \tquantum computer model."},
	{UNKNOWN, 0, "", "",option::Arg::None,
		"\nExamples:\n"
			"  snuqs -s <snuql_filename> --method=statevector --device=cpu\n"
	},

	{0,0,0,0,0,0}
};

void printUsageAndExit(int retval) {
	option::printUsage(std::cout, usage);
	std::exit(retval);
}

std::string getSnuQL(const std::vector<option::Option> &options) {
	return options[SNUQL].arg;
}

snuqs::SimulationMethod getMethod(const std::vector<option::Option> &options) {
	snuqs::SimulationMethod m = snuqs::SimulationMethod::STATEVECTOR;
	if (options[METHOD]) {
		const std::string &method = options[METHOD].arg;

		if (method == "statevector") {
			m = snuqs::SimulationMethod::STATEVECTOR;
		} else if (method == "density") {
			m = snuqs::SimulationMethod::DENSITY;
		} else if (method == "contraction") {
			m = snuqs::SimulationMethod::CONTRACTION;
		} else {
			printUsageAndExit(EXIT_FAILURE);
		}
	}
	return m;
}

snuqs::SimulationDevice getDevice(const std::vector<option::Option> &options) {
	snuqs::SimulationDevice m = snuqs::SimulationDevice::CPU;
	if (options[DEVICE]) {
		const std::string &device = options[DEVICE].arg;

		if (device == "cpu") {
			m = snuqs::SimulationDevice::CPU;
		} else if (device == "gpu") {
			m = snuqs::SimulationDevice::GPU;
		}
	}
	return m;
}


void pretty_print( std::ostream& os, boost::json::value const& jv, std::string* indent = nullptr )
{
	std::string indent_;
	if(! indent)
		indent = &indent_;
	switch(jv.kind())
	{
		case boost::json::kind::object:
			{
				os << "{\n";
				indent->append(4, ' ');
				auto const& obj = jv.get_object();
				if(! obj.empty())
				{
					auto it = obj.begin();
					for(;;)
					{
						os << *indent << boost::json::serialize(it->key()) << " : ";
						pretty_print(os, it->value(), indent);
						if(++it == obj.end())
							break;
						os << ",\n";
					}
				}
				os << "\n";
				indent->resize(indent->size() - 4);
				os << *indent << "}";
				break;
			}

		case boost::json::kind::array:
			{
				os << "[\n";
				indent->append(4, ' ');
				auto const& arr = jv.get_array();
				if(! arr.empty())
				{
					auto it = arr.begin();
					for(;;)
					{
						os << *indent;
						pretty_print( os, *it, indent);
						if(++it == arr.end())
							break;
						os << ",\n";
					}
				}
				os << "\n";
				indent->resize(indent->size() - 4);
				os << *indent << "]";
				break;
			}

		case boost::json::kind::string:
			{
				os << boost::json::serialize(jv.get_string());
				break;
			}

		case boost::json::kind::uint64:
			os << jv.get_uint64();
			break;

		case boost::json::kind::int64:
			os << jv.get_int64();
			break;

		case boost::json::kind::double_:
			os << jv.get_double();
			break;

		case boost::json::kind::bool_:
			if(jv.get_bool())
				os << "true";
			else
				os << "false";
			break;

		case boost::json::kind::null:
			os << "null";
			break;
	}

	if(indent->empty())
		os << "\n";
}


snuqs::Model getModel(const std::vector<option::Option> &options) {
	if (options[MODEL]) {
		const std::string &filename = options[MODEL].arg;

		std::ifstream f(filename);
		boost::json::stream_parser p;
		boost::json::error_code ec;
		f.seekg (0, f.end);
		int length = f.tellg();
		f.seekg (0, f.beg);
		std::string buf('0', length);
		f.read((char*)buf.c_str(), length);
		p.write((char*)buf.c_str(), length, ec);
		p.finish();
		boost::json::value value = p.release();
		pretty_print(std::cout, value);

		return snuqs::Model::NOISY;
	}
	return snuqs::Model::IDEAL;
}

bool getVerbose(const std::vector<option::Option> &options) {
	if (options[VERBOSE]) {
		return true;
	}
	return false;
}

bool getUseIO(const std::vector<option::Option> &options) {
	if (options[USEIO]) {
		return true;
	}
	return false;
}


snuqs::Job::shared_ptr
createJob(const std::vector<snuqs::QuantumCircuit> &circs, snuqs::Simulator::shared_ptr sim) {
	return std::make_unique<snuqs::Job>(circs, sim);
}

int main(int argc, char* argv[])
{
	int num_args = argc-1;
	char **argp = &argv[1];

	// -------------------
	// Parse options
	// -------------------
	option::Stats stats(usage, num_args, argp);
	std::vector<option::Option> options(stats.options_max);
	std::vector<option::Option> buffer(stats.buffer_max);
	option::Parser args(usage, num_args, argp, &options[0], &buffer[0]);

	if (args.error()) {
		exit(EXIT_FAILURE);
	}

	if (options[UNKNOWN]) {
		for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next()) {
			std::cerr << "Unknown option: " << std::string(opt->name,opt->namelen) << "\n";
		}
		std::exit(EXIT_FAILURE);
	}

	if (args.nonOptionsCount() > 0 ){
		for (int i = 0; i < args.nonOptionsCount(); ++i) {
			std::cerr << "Unknown input: "<< args.nonOption(i) << "\n";
		}
		std::exit(EXIT_FAILURE);
	}

	if (options[HELP]) {
		printUsageAndExit(EXIT_SUCCESS);
	}

	if (!options[SNUQL]) {
		printUsageAndExit(EXIT_FAILURE);
	}

	// -------------------

	//
	// Configure parameters
	//

	// TODO

	std::cout << "Simulating " << options[SNUQL].arg << "\n";
	try {
		snuqs::SnuQLCompiler compiler;
		snuqs::SnuQLOptimizer optimizer;

		//
		// 1. Compile
		//
		std::string snuql = getSnuQL(options);
		compiler.compile(snuql);

		//
		// 1.5. Analyzer
		//
		bool verbose = getVerbose(options);
		if (verbose) {
			snuqs::CircuitAnalyzer analyzer;
			analyzer.analyze(compiler.getQuantumCircuit());
		}

		//
		// 2. Transpile
		//
		snuqs::Model model = getModel(options);
		snuqs::SimulationMethod method = getMethod(options);
		snuqs::SnuQLTranspiler transpiler(method);
		transpiler.transpile(compiler.getQuantumCircuit());

		//
		// 3. Optimize
		//
		snuqs::SimulationDevice device = getDevice(options);
		bool useio = getUseIO(options);
		optimizer.optimize(transpiler.getQuantumCircuit());
		const std::vector<snuqs::QuantumCircuit> &circs = optimizer.getQuantumCircuits();
		snuqs::Simulator::shared_ptr sim = snuqs::getSimulator(method, device, useio, circs[0].num_qubits());

		//
		// 4. Simulate
		//
		snuqs::Job::shared_ptr job = createJob(circs, sim);
		job->run();
	} catch (std::exception &e) {
		std::cerr << e.what() << "\n";
		std::exit(EXIT_FAILURE);
	}

	return 0;
}
