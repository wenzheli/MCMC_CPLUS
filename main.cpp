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
#include "mcmc_sampler_stochastic_state_transition.h"


#include <time.h>

#include "global.h"

int main(int argc, char *argv[]) {	
	
	Options args;
	args.alpha = 1;
	args.eta0 = 1;                                                                                         
	args.eta1 = 1;
	args.K = 100;
	args.mini_batch_size = 50;
	args.max_iteration = 1500000;
	args.epsilon = 0.0000001;
	args.a = 0.01;
	args.b = 1024;
	args.c = 0.55;
	args.dataset_class = "tiny";

	mcmc::preprocess::DataFactory df(args.dataset_class, args.filename);
	const mcmc::Data *data = df.get_data();
	mcmc::Network network(data, 0.01); 

	args.K = 50;
	args.alpha = 0.05;

	//mcmc::learning::MCMCSamplerStochastic sampler(args, network);
	//sampler.run();
	//mcmc::learning::MCMCSamplerBatch sampler1(args, network);
	//sampler1.run();	

	//mcmc::learning::GibbsSampler sampler2(args, network);
	//sampler2.run();		

/*
	int* kk =  new int[15];
	kk[0] = 2;
	kk[1] = 3;
	kk[2] = 5;
	kk[3] = 7; 
	kk[4] = 9;
	kk[5] = 10;
	kk[6] = 15;
	kk[7] = 20;
	kk[8] = 30;
	kk[9] = 50;
	kk[10] = 70;
	kk[11] = 80;
	kk[12] = 100;
	kk[13] = 120;
	kk[14] = 150;

	for (int i = 0; i < 15; i++){
		args.K = kk[i];
		args.alpha = 1.0/args.K;
		mcmc::learning::MCMCSamplerStochastic sampler(args, network);
		sampler.run();
		mcmc::learning::SGD sampler1 (args, network);
		sampler1.run();
	}
	*/
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

	
	network.set_num_pieces(20);
	mcmc::learning::MCMCSamplerStochasticStateTransition sampler1(args, network);
	sampler1.setNumNodeSample(20);
	sampler1.run();
	
	/*
	network.set_num_pieces(20);
	mcmc::learning::MCMCSamplerStochastic sampler2(args, network);
	sampler2.setNumNodeSample(20);
	sampler2.run();

	network.set_num_pieces(30);
	mcmc::learning::MCMCSamplerStochastic sampler3(args, network);
	sampler3.setNumNodeSample(30);
	sampler3.run();

	network.set_num_pieces(40);
	mcmc::learning::MCMCSamplerStochastic sampler4(args, network);
	sampler4.setNumNodeSample(40);
	sampler4.run();

	network.set_num_pieces(50);
	mcmc::learning::MCMCSamplerStochastic sampler5(args, network);
	sampler5.setNumNodeSample(50);
	sampler5.run();

	network.set_num_pieces(70);
	mcmc::learning::MCMCSamplerStochastic sampler6(args, network);
	sampler6.setNumNodeSample(70);
	sampler6.run();

	network.set_num_pieces(100);
	mcmc::learning::MCMCSamplerStochastic sampler7(args, network);
	sampler7.setNumNodeSample(100);
	sampler7.run();

	
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



