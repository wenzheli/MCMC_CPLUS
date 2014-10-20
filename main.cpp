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

#include <time.h>

#include "global.h"

int main(int argc, char *argv[]) {	


	Options args;
	args.alpha = 0.02;
	args.eta0 = 1;                                                                                         
	args.eta1 = 1;
	args.K = 50;
	args.mini_batch_size = 50;
	args.max_iteration = 200000;
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
	/*
	for (int k = 10; k < 110; k=k+10){
		clock_t t1, t2;
		t1 = clock();
		args.K = k;
		mcmc::learning::MCMCSamplerStochastic sampler(args, network);
		sampler.run();

		t2 = clock();
		float diff = ((float)t2 - (float)t1);
		float seconds = diff / CLOCKS_PER_SEC;
		cout << seconds << endl;
	}
	*/
	
	return 1;																																	
}	



