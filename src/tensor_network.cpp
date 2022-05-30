#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <ostream>

#include "tensor.hpp"
#include "circuit.hpp"
#include "edge.hpp"

namespace thunder {

class TensorNetwork {
    public:

    TensorNetwork(const Circuit &circuit)
    : _num_qubits(circuit.num_qubits()) {
		_grid_map.resize(_num_qubits);

		for (size_t i = 0; i < _num_qubits; i++) {
			_grid_map[i] = 0;
		}

		_num_amps = 0;
		_max_cycle = 0;
		for (const auto &g : circuit.gates()) {
			size_t cycle = 0;

			auto qubits = g->qubits();
			
			for (auto q : qubits) {
				if (cycle < _grid_map[q]+1) {
					cycle = _grid_map[q]+1;
				}
			}

			if (qubits.size() == 1) {
				std::vector<Edge> edges;
				for (auto q : qubits) {
					edges.emplace_back(CreateEdge(_grid_map[q], q));
				}

				for (auto q : qubits) {
					edges.emplace_back(CreateEdge(cycle, q));
				}

				std::string name = thunder::utils::coord_string(cycle, qubits[0]);
				_tensor_list.emplace_back(name, edges, tensor::GetData<F>(gate::type_to_string(g->type()), g->params()));

				_num_amps += (1ULL << edges.size());
			} 
			else {
				std::vector<Edge> edges1;
				std::vector<Edge> edges2;

				edges1.emplace_back(CreateEdge(_grid_map[qubits[0]], qubits[0]));
				edges1.emplace_back(CreateEdge(_grid_map[qubits[0]], qubits[1]));
				edges1.emplace_back(CreateEdge(cycle, qubits[0]));

				edges2.emplace_back(CreateEdge(_grid_map[qubits[1]], qubits[1]));
				edges2.emplace_back(CreateEdge(_grid_map[qubits[0]], qubits[1]));
				edges2.emplace_back(CreateEdge(cycle, qubits[1]));

				std::string name1 = thunder::utils::coord_string(cycle, qubits[0]);
				std::string name2 = thunder::utils::coord_string(cycle, qubits[1]);

				std::vector<double> data1 = {1., 0., 0., 0., 0., 0., 0., 1.};
				std::vector<double> data2 = {1., 0., 1., 0., 0., 1., 0., -1.};
				_tensor_list.emplace_back(name1, edges1, data1);
				_tensor_list.emplace_back(name2, edges2, data2);

				_num_amps += (1ULL << edges1.size());
				_num_amps += (1ULL << edges2.size());
			}

			for (auto q : qubits) {
				_grid_map[q] = cycle;
			}
		}
	}

    TensorNetwork(const TensorNetwork &other) 
    : _tensor_list(other.tensor_list()),
      _index_map(other.index_map()),
      _grid_map(other.grid_map()),
      _num_qubits(other.num_qubits()),
      _num_amps(other.num_amps()),
      _max_cycle(other.max_cycle())
    { }


    TensorNetwork(TensorNetwork &&other)
    : _tensor_list(std::move(other.tensor_list())),
      _index_map(std::move(other.index_map())),
      _grid_map(std::move(other.grid_map())),
      _num_qubits(std::move(other.num_qubits())),
      _num_amps(std::move(other.num_amps())),
      _max_cycle(std::move(other.max_cycle()))
    { }

    TensorNetwork& operator=(const TensorNetwork &other) {
		if (this != &other) {
			_tensor_list = other.tensor_list();
			_index_map = other.index_map();
			_grid_map = other.grid_map();
			_num_qubits = other.num_qubits();
			_num_amps = other.num_amps();
			_max_cycle = other.max_cycle();
		}
    	return *this;
	}

    TensorNetwork& operator=(TensorNetwork &&other) {
		if (this != &other) {
			_tensor_list = std::move(other.tensor_list());
			_index_map = std::move(other.index_map());
			_grid_map = std::move(other.grid_map());
			_num_qubits = std::move(other.num_qubits());
			_num_amps = std::move(other.num_amps());
			_max_cycle = std::move(other.max_cycle());
		}
    	return *this;
	}

    const std::vector<Tensor<F>>& tensor_list() const { return _tensor_list; }

    const IndexMap& index_map() const { return _index_map; }

    const std::vector<size_t>& grid_map() const { return _grid_map; }

    size_t num_qubits() const { return _num_qubits; }

    size_t num_amps() const { return _num_amps; }

    size_t max_cycle() const { return _max_cycle; }
    
    void Insert(const Tensor<F> &tensor) {
        _tensor_list.push_back(tensor);
        _num_amps += tensor.GetNumAmplitudes();
    }

    Tensor<F>& Find(const std::string &name) {
        for (auto it = _tensor_list.begin(); it != _tensor_list.end(); it++) {
            if (it->name() == name) {
                return *it;
            }
        }
        throw error::runtime("Cannot find tensor ", string::quote(name));
    }
    
    Tensor<F>& FindTensorWithEdge(const std::string &name, size_t min_dim=2) {
		std::string f = _index_map.FindValue(name);
        for (auto it = _tensor_list.begin(); it != _tensor_list.end(); it++) {
			if (it->edges().size() >= min_dim) {
				for (auto e : it->edges()) {
					if (e.index() == f)
						return *it;
				}
			}
        }
        throw error::runtime("Cannot find tensor with edge", string::quote(name));
	}

    void Erase(const std::vector<std::string> &names) {
        // FIXME: DO NOT USE REFERENCE
        for (auto it = _tensor_list.begin();
                it != _tensor_list.end();
                ) {
            if ((it->name() == names[0]) || (it->name() == names[1])) {
				_num_amps -= it->GetNumAmplitudes();
                it = _tensor_list.erase(it);
            } else {
                it++;
            }
        }
    }

    size_t GetNumAmplitudes() const {
    	return _num_amps;
	}

	void AddInputTheta0(size_t i) {
		std::string name = "#i" + std::to_string(i);
		std::string idx = thunder::utils::coord_string(0, i);
		_tensor_list.emplace_back(idx, Edge(_index_map.FindIndex(idx)),
				tensor::GetData<F>("THETA0"));
	}

	void AddInputTheta1(size_t i) {
		std::string name = "#i" + std::to_string(i);
		std::string idx = thunder::utils::coord_string(0, i);
		_tensor_list.emplace_back(idx, Edge(_index_map.FindIndex(idx)),
				tensor::GetData<F>("THETA1"));
	}

	void AddOutputTheta0(size_t i) {
		std::string name = thunder::utils::coord_string(_grid_map[i]+1, i);
		std::string idx = thunder::utils::coord_string(_grid_map[i], i);
		_tensor_list.emplace_back(name, Edge(_index_map.FindIndex(idx)),
				tensor::GetData<F>("THETA0"));
	}

	void AddOutputTheta1(size_t i) {
		std::string name = thunder::utils::coord_string(_grid_map[i]+1, i);
		std::string idx = thunder::utils::coord_string(_grid_map[i], i);
		_tensor_list.emplace_back(name, Edge(_index_map.FindIndex(idx)),
				tensor::GetData<F>("THETA1"));
	}

	void Sort() {
		std::sort(_tensor_list.begin(), _tensor_list.end(), 
				[](auto &a, auto &b) {
				std::vector<std::string> tokensA;
				std::string nameA = a.name().substr(1, a.name().length()-2);
				boost::split(tokensA, nameA, boost::is_any_of(", "));
				std::string nameB = b.name().substr(1, b.name().length()-2);
				std::vector<std::string> tokensB;
				boost::split(tokensB, nameB, boost::is_any_of(", "));

				return (std::stoul(tokensA[0]) < std::stoul(tokensB[0])) ||
				((std::stoul(tokensA[0]) == std::stoul(tokensB[0])) &&
				 (std::stoul(tokensA[1]) < std::stoul(tokensB[1])));
					}
				);
	}

    std::vector<Tensor<F>>& tensor_list() {
        return _tensor_list;
    }

    Edge CreateEdge(size_t cycle, size_t qubit, size_t dim=2) {
        std::string idx = thunder::utils::coord_string(cycle, qubit);
        return Edge(_index_map.FindOrCreateIndex(idx), 2);
    }

    template<typename T>
    friend std::ostream& operator<<(std::ostream &os, const TensorNetwork<T> &net);

    std::ostream& operator<<(std::ostream &os) const {
        for (auto &t : _tensor_list)
            std::cout << "Tensor_" << t << "\n";
        return os;
    }

    private:

    std::vector<Tensor<F>> _tensor_list;
    IndexMap _index_map;
    std::vector<size_t> _grid_map;
    size_t _num_qubits;
    size_t _num_amps;
    size_t _max_cycle;

};

template<typename T>
std::ostream &operator<<(std::ostream &os, const TensorNetwork<T> &net) {
    return net.operator<< (os);
}

}  // namespace thunder
