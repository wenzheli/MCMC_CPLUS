#ifndef MCMC_PREPROCESS_DATA_FACTORY_H__
#define MCMC_PREPROCESS_DATA_FACTORY_H__

#include "data.h"
#include "dataset.h"
#include "relativity.h"
#include "tiny.h"

namespace mcmc {
	namespace preprocess {

		class DataFactory {
		public:
			DataFactory(const std::string &dataset_name, const std::string &filename = "")
				: dataset_name(dataset_name), filename(filename) {
			}

			virtual ~DataFactory() {
			}

			const mcmc::Data *get_data() const {
				DataSet *dataObj = NULL;
				if (false) {
				}
				else if (dataset_name == "tiny"){
					dataObj = new TinyNetwork(filename);
				}
				else if (dataset_name == "relativity") {
					dataObj = new Relativity(filename);
				}
				else if (dataset_name == "hep_ph") {
					// dataObj = new HepPH(filename);
				}
				else if (dataset_name == "astro_ph") {
					// dataObj = new AstroPH(filename);
				}
				else if (dataset_name == "condmat") {
					// dataObj = new CondMat(filename);
				}
				else if (dataset_name == "hep_th") {
					// dataObj = new HepTH(filename);
				}
				else {
					throw MCMCException("Unknown dataset name \"" + dataset_name + "\"");
				}

				return dataObj->process();
			}

		protected:
			std::string dataset_name;
			std::string filename;
		};

	};	// namespace preprocess
};	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_DATA_FACTORY_H__
