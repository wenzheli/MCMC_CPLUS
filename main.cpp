#include <iostream>
#include "data.h"
#include "data_factory.h"
#include "dataset.h"
#include "exception.h"
#include "learner.h"
#include "mcmc.h"
#include "network.h"
#include "np.h"
#include "myrandom.h"
#include "relativity.h"
#include "types.h"
#include "mcmc_sampler_batch.h"
#include "mcmc_sampler_stochastic.h"
#include "variational_inference_stochastic.h"
#include "gibbs_sampler.h"
#include "mcmc_sgd.h"


#include <time.h>

#include "global.h"

int main(int argc, char *argv[]) {	


	Options args;
	args.alpha = 1;
	args.eta0 = 1;                                                                                         
	args.eta1 = 1;
	args.K = 10;
	args.mini_batch_size = 50;
	args.max_iteration = 5000000;
	args.epsilon = 0.0000001;
	args.a = 0.01;
	args.b = 1024;
	args.c = 0.55;
	args.dataset_class = "relativity";

	mcmc::preprocess::DataFactory df(args.dataset_class, args.filename);
	const mcmc::Data *data = df.get_data();
	mcmc::Network network(data, 0.01); 


	mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	sampler.run();

	mcmc::learning::SGD sampler1(args, network);
	sampler1.run();

	
	
	/*
	for (int k = 150; k < 160; k=k+10){
		clock_t t1, t2;
		t1 = clock();
		args.K = k;
		args.alpha = 1.0/args.K;
		mcmc::learning::MCMCSamplerStochastic sampler(args, network);
		sampler.run();

		t2 = clock();
		float diff = ((float)t2 - (float)t1);
		float seconds = diff / CLOCKS_PER_SEC;
		cout << seconds << endl;
	}
	*/
	/*
	args.K = 180;
	args.alpha = 1.0/args.K;
	mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	sampler.run();
*/
	
	return 1;																																	
}	



