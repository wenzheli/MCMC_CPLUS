#ifndef MCMC_PREPROCESS_TINY_H__
#define MCMC_PREPROCESS_TINY_H__

#include "data.h"
#include "dataset.h"
#include <algorithm>
#include <set>
#include <fstream>


using namespace std;
namespace mcmc {
	namespace preprocess {

		// FIXME: identical: hep_ph relativity ...

		/**
		* Process relativity data set
		*/
		class TinyNetwork : public DataSet {

		public:
			const int MAX_NODES = 100;

		public:
			TinyNetwork(const std::string &filename) : DataSet(filename == "" ? "Tiny.txt" : filename) {
			}

			virtual ~TinyNetwork() {
			}

			/**
			* The data is stored in .txt file. The format of data is as follows, the first column
			* is line number. Within each line, it is tab separated.
			*
			* [1] 1    100
			* [2] 1    103
			* [3] 4    400
			* [4] ............
			*
			* However, the node ID is not increasing by 1 every time. Thus, we re-format
			* the node ID first.
			*/
			virtual const Data *process() {
				std::ifstream infile(filename);
				if (!infile) {
					throw mcmc::IOException("Cannot open " + filename);
				}

				std::string line;
				
				// start from the 5th line.
				std::set<int> vertex;	// ordered set
				std::vector<mcmc::Edge> edge;
				while (std::getline(infile, line)) {
					int a;
					int b;
					std::istringstream iss(line);
					if (!(iss >> a >> b)) {
						throw mcmc::IOException("Fail to parse int");
					}
					vertex.insert(a);
					vertex.insert(b);
					edge.push_back(Edge(a, b));
				}

				std::vector<int> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

				::size_t N = nodelist.size();

				// change the node ID to make it start from 0
				std::unordered_map<int, int> node_id_map;
				int i = 0;
				for (std::vector<int>::iterator node_id = nodelist.begin();
					node_id != nodelist.end();
					node_id++) {
					node_id_map[*node_id] = i;
					i++;
				}

				mcmc::EdgeSet *E = new mcmc::EdgeSet();	// store all pair of edges.
				for (std::vector<Edge>::iterator i = edge.begin();
					i != edge.end();
					i++) {
					int node1 = node_id_map[i->first];
					int node2 = node_id_map[i->second];
					if (node1 == node2) {
						continue;
					}

					if (node1 >= MAX_NODES || node2 >= MAX_NODES){
						continue;
					}

					E->insert(Edge(std::min(node1, node2), std::max(node1, node2)));
				}

				//N = MAX_NODES;

				return new Data(NULL, E, N);
			}

		};

	}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_RELATIVITY_H__
