#ifndef MCMC_LEARNING_LEARNER_H__
#define MCMC_LEARNING_LEARNER_H__

#include <cmath>

#include "options.h"
#include "types.h"
#include "network.h"

namespace mcmc {
	namespace learning {

		/**
		* This is base class for all concrete learners, including MCMC sampler, variational
		* inference,etc.
		*/
		class Learner {
		public:
			/**
			* initialize base learner parameters.
			*/
			Learner(const Options &args, const Network &network)
				: network(network) {

				// model priors
				alpha = args.alpha;
				eta = new double[2];
				eta[0] = args.eta0;
				eta[1] = args.eta1;
				average_count = 1;

				// parameters related to control model
				K = args.K;
				epsilon = args.epsilon;

				// parameters related to network
				N = network.get_num_nodes();

				// model parameters to learn
				//beta = std::vector<double>(K, 0.0);
				//pi = std::vector<std::vector<double> >(N, std::vector<double>(K, 0.0));
				beta = new double[K]();
				for (int k = 0; k < K; k++){
					beta[k] = 0;
				}
				cout << beta[0];

				pi = new double*[N];
				for (int i = 0; i < N; i++){
					pi[i] = new double[K];
				}

				cout << pi[0][0];


				// parameters related to sampling
				mini_batch_size = args.mini_batch_size;
				if (mini_batch_size < 1) {
					mini_batch_size = N / 2;	// default option.
				}

				// ration between link edges and non-link edges
				link_ratio = network.get_num_linked_edges() / ((N * (N - 1)) / 2.0);
				// check the number of iterations.
				step_count = 1;
				// store perplexity for all the iterations
				// ppxs_held_out = [];
				// ppxs_test = [];

				max_iteration = args.max_iteration;
				CONVERGENCE_THRESHOLD = 0.000000000001;

				stepsize_switch = false;

				ppx_for_heldout = new double[network.held_out_size]();
			}

			virtual ~Learner() {
			}

			/**
			* Each concrete learner should implement this. It basically
			* iterate the data sets, then update the model parameters, until
			* convergence. The convergence can be measured by perplexity score.
			*
			* We currently support four different learners:
			* 1. MCMC for batch learning
			* 2. MCMC for mini-batch training
			* 3. Variational inference for batch learning
			* 4. Stochastic variational inference
			*/
			virtual void run() = 0;

		protected:
			const std::vector<double> &get_ppxs_held_out() const {
				return ppxs_held_out;
			}

			const std::vector<double> &get_ppxs_test() const {
				return ppxs_test;
			}

			void set_max_iteration(int max_iteration) {
				this->max_iteration = max_iteration;
			}

			double cal_perplexity_held_out() {
				cout<<"calling cal_perplexity_held_out";
				return cal_perplexity(network.get_held_out_set());
			}

			double cal_perplexity_test() {
				return cal_perplexity(network.get_test_set());
			}

			bool is_converged() const {
				int n = ppxs_held_out.size();
				if (n < 2) {
					return false;
				}
				if (std::abs(ppxs_held_out[n - 1] - ppxs_held_out[n - 2]) / ppxs_held_out[n - 2] >
					CONVERGENCE_THRESHOLD) {
					return false;
				}

				return true;
			}


		protected:
			/**
			* calculate the perplexity for data.
			* perplexity defines as exponential of negative average log likelihood.
			* formally:
			*     ppx = exp(-1/N * \sum){i}^{N}log p(y))
			*
			* we calculate average log likelihood for link and non-link separately, with the
			* purpose of weighting each part proportionally. (the reason is that we sample
			* the equal number of link edges and non-link edges for held out data and test data,
			* which is not true representation of actual data set, which is extremely sparse.
			*/
			double cal_perplexity(const EdgeMap &data) {
				double link_likelihood = 0.0;
				double non_link_likelihood = 0.0;
			
			
				int link_count = 0;
				int non_link_count = 0;
				cout<<ppx_for_heldout[0]<<endl;
				int idx = 0;
				for (EdgeMap::const_iterator edge = data.begin();
					edge != data.end();
					edge++) {
					
					const Edge &e = edge->first;
					double edge_likelihood = cal_edge_likelihood(pi[e.first], pi[e.second],
						edge->second, beta);
					//cout<<edge_likelihood;
					//cout<<"edge likelihood"<<edge_likelihood<<endl;

					if (std::isnan(edge_likelihood)){
						cout<<"potential bug";
					}

					//cout<<"AVERAGE COUNT: " <<average_count;
					ppx_for_heldout[idx] = (ppx_for_heldout[idx] * (average_count-1) + edge_likelihood)/(average_count);

					//cout<<ppx_for_heldout[idx];
					// std::cerr << std::fixed << std::setprecision(12) << e << " in? " << (e.in(network.get_linked_edges()) ? "True" : "False") << " -> " << edge_likelihood << std::endl;
					if (e.in(network.get_linked_edges())) {
						link_count++;
						link_likelihood += std::log(ppx_for_heldout[idx]);
						//link_likelihood += edge_likelihood;

						if (std::isnan(link_likelihood)){
							cout<<"potential bug";
						}
					}
					else {
						assert(!present(network.get_linked_edges(), e));
						non_link_count++;
						//non_link_likelihood += edge_likelihood;
						non_link_likelihood += std::log(ppx_for_heldout[idx]);
						if (std::isnan(non_link_likelihood)){
							cout<<"potential bug";
						}
					}
					idx++;
				}
				// std::cerr << std::setprecision(12) << "ratio " << link_ratio << " count: link " << link_count << " " << link_likelihood << " non-link " << non_link_count << " " << non_link_likelihood << std::endl;

				// weight each part proportionally.
				// avg_likelihood = self._link_ratio*(link_likelihood/link_count) + \
						//         (1-self._link_ratio)*(non_link_likelihood/non_link_count)

				// direct calculation.
				double avg_likelihood = 0.0;
				if (link_count + non_link_count != 0){
					avg_likelihood = (link_likelihood + non_link_likelihood) / (link_count + non_link_count);
				}
				if (true) {
					double avg_likelihood1 = link_ratio * (link_likelihood / link_count) + \
						(1.0 - link_ratio) * (non_link_likelihood / non_link_count);
					std::cerr << std::setprecision(12) << avg_likelihood << " " << (link_likelihood / link_count) << " " << link_count << " " << \
						(non_link_likelihood / non_link_count) << " " << non_link_count << " " << avg_likelihood1 << std::endl;
					// std::cerr << "perplexity score is: " << exp(-avg_likelihood) << std::endl;
				}

				// return std::exp(-avg_likelihood);

				//if (step_count > 1000)
				average_count = average_count + 1;
				cout<<"average_count is: "<<average_count;
				return (-avg_likelihood);
			}


			template <typename T>
			static void dump(const std::vector<T> &a, int n, const std::string &name = "") {
				n = std::min(n, a.size());
				std::cerr << name;
				if (n != a.size()) {
					std::cerr << "[0:" << n << "]";
				}
				std::cerr << " ";
				for (auto i = a.begin(); i < a.begin() + n; i++) {
					std::cerr << std::setprecision(12) << *i << " ";
				}
				std::cerr << std::endl;
			}


			/**
			* calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
			* in order to calculate this, we need to sum over all the possible (z_ab, z_ba)
			* such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
			* but this calculation can be done in O(K), by using some trick.
			*/
			double cal_edge_likelihood(double* pi_a,
				double* pi_b,
				bool y,
				const double* beta) const {
				double s = 0.0;
				if (y) {
					for (int k = 0; k < K; k++) {
						s += pi_a[k] * pi_b[k] * beta[k];
					}
				}
				else {
					double sum = 0.0;
					for (int k = 0; k < K; k++) {
						// FIXME share common subexpressions
						s += pi_a[k] * pi_b[k] * (1.0 - beta[k]);
						sum += pi_a[k] * pi_b[k];
					}
					s += (1.0 - sum) * (1.0 - epsilon);
				}

				if (s < 1.0e-30) {
					s = 1.0e-30;
				}

				return s;
#if 0
				double prob = 0.0;
				double s = 0.0;

				for (int k = 0; k < K; k++) {
					if (!y) {
						prob += pi_a[k] * pi_b[k] * (1 - beta[k]);
					}
					else {
						prob += pi_a[k] * pi_b[k] * beta[k];
					}
					s += pi_a[k] * pi_b[k];		// common expr w/ above
				}

				if (!y) {
					prob += (1.0 - s) * (1 - epsilon);
				}
				else {
					prob += (1.0 - s) * epsilon;
				}
				// std::cerr << "Calculate s " << s << " prob " << prob << std::endl;
				if (prob < 0.0) {
					std::cerr << "adsfadsfadsf" << std::endl;
				}

				return log(prob);
#endif
			}

		protected:
			const Network &network;

			double alpha;
			double* eta;
			int K;
			double epsilon;
			int N;

			//std::vector<double> beta;
			//std::vector<std::vector<double>> pi;
			double* beta;
			double** pi;

			int mini_batch_size;
			double link_ratio;

			int step_count;

			std::vector<double> ppxs_held_out;
			std::vector<double> ppxs_test;
			std::vector<double> iterations;
			double* ppx_for_heldout;

			int max_iteration;

			double CONVERGENCE_THRESHOLD;

			bool stepsize_switch;
			int average_count;
		};


	}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_LEARNER_H__
