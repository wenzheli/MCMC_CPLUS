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
	args.K = 30;
	args.mini_batch_size = 50;
	args.max_iteration = 3000000;
	args.epsilon = 0.0000001;
	args.a = 0.01;
	args.b = 1024;
	args.c = 0.55;
	args.dataset_class = "relativity";

	mcmc::preprocess::DataFactory df(args.dataset_class, args.filename);
	const mcmc::Data *data = df.get_data();
	mcmc::Network network(data, 0.01); 

	/*
	args.K = 20;
	args.alpha = 0.01;

	mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	sampler.run();

	args.alpha = 0.05;
	mcmc::learning::MCMCSamplerStochastic sampler1(args, network);
	sampler1.run();

	args.alpha = 0.1;
	mcmc::learning::MCMCSamplerStochastic sampler2(args, network);
	sampler2.run();

	args.alpha = 0.5;
	mcmc::learning::MCMCSamplerStochastic sampler3(args, network);
	sampler3.run();

	args.alpha = 1;
	mcmc::learning::MCMCSamplerStochastic sampler4(args, network);
	sampler4.run();
*/

	
	args.K = 15;
	args.alpha = 0.01;
	mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	//sampler.run();
	network.set_num_pieces(10);
	sampler.setNumNodeSample(10);
	sampler.run();

	mcmc::learning::MCMCSamplerStochastic sampler1(args, network);
	network.set_num_pieces(20);
	sampler1.setNumNodeSample(20);
	sampler1.run();

	mcmc::learning::MCMCSamplerStochastic sampler2(args, network);
	network.set_num_pieces(30);
	sampler2.setNumNodeSample(30);
	sampler2.run();

	mcmc::learning::MCMCSamplerStochastic sampler3(args, network);
	network.set_num_pieces(50);
	sampler3.setNumNodeSample(50);
	sampler3.run();

	mcmc::learning::MCMCSamplerStochastic sampler4(args, network);
	network.set_num_pieces(70);
	sampler4.setNumNodeSample(70);
	sampler4.run();

	mcmc::learning::MCMCSamplerStochastic sampler5(args, network);
	network.set_num_pieces(100);
	sampler5.setNumNodeSample(100);
	sampler5.run();

	mcmc::learning::MCMCSamplerStochastic sampler6(args, network);
	network.set_num_pieces(150);
	sampler6.setNumNodeSample(150);
	sampler6.run();



	
	//mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	//sampler.run();

	//mcmc::learning::SGD sampler1(args, network);
	//sampler1.run();

	
	/*
	for (int k = 10; k < 110; k=k+10){
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



