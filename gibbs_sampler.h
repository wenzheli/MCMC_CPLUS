#ifndef GIBBS_SAMPLER_H
#define GIBBS_SAMPLER_H


#include <cassert>
#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "myrandom.h"
#include "learner.h"
#include "myrandom.h"

namespace mcmc {
	  namespace learning{
	
class GibbsSampler : public Learner {


public:	
	GibbsSampler(const Options &args, const Network &graph)
		: Learner(args, graph){

		// allocate memory for variables.
		z = new int*[N]; 
		for (int i = 0; i < N; i++){
			z[i] = new int[N]();
		}

		num_kk = new int*[K];
		for (int i = 0; i < K; i++){
			num_kk[i] = new int[2]();
		}

		num_n_k = new int*[N];
		for (int i = 0; i < N; i++){
			num_n_k[i] = new int[K]();
		}

		initialize();
	}

	void initialize(){
		for (int i = 0; i < N; i++){
			for (int j = i+1; j < N; j++){
				Edge edge(i,j);
				if (edge.in(network.get_held_out_set())){
							continue;
				}
				int y = 0;
				if (edge.in(network.get_linked_edges())) {
					y = 1;
				}

				z[i][j] = Random::random->randint(0, K);
				z[j][i] = Random::random->randint(0, K);

				num_n_k[i][z[i][j]] += 1;
				num_n_k[j][z[j][i]] += 1;

				if (z[i][j] == z[j][i]){
					if (y == 1){
						num_kk[z[i][j]][0] += 1;
					}else{
						num_kk[z[i][j]][1] += 1;
					}
				}
			}
		}
	}

	void run(){
		while (step_count < max_iteration && !is_converged()) {
			auto l1 = std::chrono::system_clock::now();
			//print "step: " + str(self._step_count)
			cout<<"calling hold out...";
			double ppx_score = cal_perplexity_held_out();
			std::cout << std::fixed << std::setprecision(12) << "perplexity for hold out set: " << ppx_score << std::endl;
			ppxs_held_out.push_back(ppx_score);

			process();
			update_pi_beta();				
		}
	}


	void process(){
		for (int i = 0; i < N; i++){
			for (int j = i+1; j < N; j++){
				Edge edge(i,j);
				if (edge.in(network.get_held_out_set())){
							continue;
				}
				int y = 0;
				if (edge.in(network.get_linked_edges())) {
					y = 1;
				}

				// remove current assignment
				int z_ij_old = z[i][j];
				int z_ji_old = z[j][i];

				num_n_k[i][z_ij_old] -= 1;
				num_n_k[j][z_ji_old] -= 1;

				if (z_ij_old == z_ji_old){
					if (y==1){
						num_kk[z_ij_old][0] -= 1;
					}else{
						num_kk[z_ij_old][1] -= 1;
					}
				}

				int* results = sampler(i,j,y);

				// update
				z[i][j] = results[0];
				z[j][i] = results[1];

				num_n_k[i][z[i][j]] += 1;
				num_n_k[j][z[j][i]] += 1;

				if (z[i][j] == z[j][i]){
					if (y ==1){
						num_kk[z[i][j]][0] += 1;
					}else{
						num_kk[z[i][j]][1] += 1;
					}
				}
			}
		}
	}

	int getSum(int a[], int n) const{
		int s = 0.0;
		for (int i = 0; i < n; i++){
			s += a[i];
		}
		return s;
	}

	void update_pi_beta(){
		// update pi
		for (int i = 0; i < N; i++){
			for (int j = 0; j < K; j++){
				pi[i][j] = num_n_k[i][j]/(1.0 * getSum(num_n_k[i], K));
			}
		}
		// update beta
		for (int k = 0; k < K; k++){
			beta[k] = (1+num_kk[k][0])*1.0/(num_kk[k][0] + num_kk[k][1] + 1);
		}
	}

	int* sampler(int i, int j, int y){
		double** p;
		p = new double*[K];
		for (int i = 0; i < K; i++){
			p[i] = new double[K];
		}

		double term  = 0.0;
		for (int k1 = 0; k1 < K; k1++){
			for (int k2 = 0; k2 < K; k2++){
				if (k1 != k2){
					if (y == 1){
						term = epsilon;
					}else{
						term = 1 - epsilon;
					}
					p[k1][k2] = (alpha + num_n_k[i][k1]) * (alpha + num_n_k[j][k2]) * term;
				}else{
					if (y == 1){
						term = (num_kk[k1][0] + eta[0])/(num_kk[k1][0] + num_kk[k1][1] + eta[0] + eta[1]);
					}else{
						term = (num_kk[k1][1] + eta[1])/(num_kk[k1][0] + num_kk[k1][1] + eta[0] + eta[1]);
					}
					p[k1][k2] = (alpha + num_n_k[i][k1]) * (alpha + num_n_k[j][k2]) * term;
				}
			}
		}
	
		// sample from p
		int* results = new int[2];
		int n = K * K;
		double* temp = new double[n];
		int cnt = 0;
		for (int i = 0; i < K; i++){
			for (int j = 0; j < K; j++){
				temp[cnt] = p[i][j];
				cnt += 1;
			}
		}
		for (int i = 1; i < n; i++){
			temp[i] += temp[i-1];
		}

		double u = Random::random->random() * temp[n-1];
		int idx = 0;
		for (int i = 0; i < n; i++){
			if (u <= temp[i]){
				idx = i;
				break;
			}
		}

		results[0] = idx/K;
		results[1] = idx%K;

		for (int i = 0; i < K; i++){
			delete[] p[i];
		}
		delete[] p;
		delete[] temp;

		return results;
	}



public:
	int** z;
	int** num_kk;
	int** num_n_k;
};
}
} // mcmc namespace


#endif

