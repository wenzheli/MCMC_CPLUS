#ifndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
#define MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__

#include <cassert>
#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "np.h"
#include "myrandom.h"
#include "learner.h"

using namespace std;

namespace mcmc {
	namespace learning {
		template <typename T>
		class MCMCMyOp {
		public:
			MCMCMyOp(T a, T b) : a(a), b(b) {
			}

			T operator() (const T &x, const T &y) {
				return x * a + y * b;
			}

		protected:
			T a;
			T b;
		};


		/**
		* MCMC Sampler for batch learning. Every update go through the whole data sets.
		*/
		class MCMCSamplerBatch : public Learner {

		public:
			MCMCSamplerBatch(const Options &args, const Network &graph)
				: Learner(args, graph) {

				// step size parameters. step size = (a*(1+t/b))^(-c), t is the number of steps so far
				this->a = args.a;
				this->b = args.b;
				this->c = args.c;

				// control parameters for learning
				// TODO need to find the optimal value from experiments. If the value is too small
				// 		then it will add more variance.. if too large, computational complexity....
				num_node_sample = static_cast< int>(std::sqrt(network.get_num_nodes()));

				// model parameters and re-parameterization
				// since the model parameter - \pi and \beta should stay in the simplex,
				// we need to restrict the sum of probability equals to 1.  The way we
				// restrict this is using re-reparameterization techniques, where we
				// introduce another set of variables, and update them first followed by
				// updating \pi and \beta.
				theta = Random::random->gammaArray(eta[0], eta[1], K, 2);		// parameterization for \beta - K by 2
				phi = Random::random->gammaArray(1, 1, N, K);					// parameterization for \pi   - N by K

				// FIXME RFHH -- code sharing with variational_inf*::update_pi_beta()
				// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
				// self._beta = temp[:,1]
				//std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
				//np::row_normalize(&temp, theta);
				//std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
				// self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
				//pi.resize(phi.size(), std::vector<double>(phi[0].size()));
				//np::row_normalize(&pi, phi);
				update_pi_from_phi();
				update_beta_from_theta();
			}

			void update_pi_from_phi(){
				for (int i = 0; i < N; i++){
					double sum = 0;
					for (int k = 0; k < K; k++){
						sum += phi[i][k];
					}
					for (int k = 0; k < K; k++){
						pi[i][k] = phi[i][k] / sum;
						//cout<<pi[i][k]<< " ";
					}
					//cout<<endl;
				}
			}

			void update_beta_from_theta(){
				for (int k = 0; k < K; k++){
					double sum = 0;
					for (int t = 0; t < 2; t++){
						sum += theta[k][t];
					}
					// beta[k] = theta[k][1]/(theta[k][0] + theta[k][1])
					beta[k] = theta[k][1] / sum;
				}
			}


			virtual ~MCMCSamplerBatch() {
			}


			// FIXME make VertexSet a result parameter
			OrderedVertexSet sample_neighbor_nodes_batch(int node) const {
				OrderedVertexSet neighbor_nodes;
				for (int i = 0; i < (int)N; i++) {
					Edge edge(std::min(node, i), std::max(node, i));
					if (!edge.in(network.get_held_out_set()) && !edge.in(network.get_test_set())) {
						if (false && i == node) {
							std::cerr << "Ooppssss.... is a self-cycle OK? " << i << std::endl;
						}
						neighbor_nodes.insert(i);
					}
				}

				return neighbor_nodes;
			}


			double getSum(double a[], int n) const{
				double s = 0.0;
				for (int i = 0; i < n; i++){
					s += a[i];
				}
				return s;
			}


			/**
			* update pi for current node i.
			could leave this function as it is, but we don't use this for now... [wenzhe]
			*/
			void update_pi_for_node(int i, const std::vector<int> &z, std::vector<std::vector<double> > *phi_star, int n) const {
				// update gamma, only update node in the grad
				double eps_t;

				if (!stepsize_switch) {
					eps_t = std::pow(1024 + step_count, -0.5);
				}
				else {
					eps_t = a * std::pow(1 + step_count / b, -c);
				}

				double phi_i_sum = getSum(phi[i], K);
				double* noise = Random::random->randnArray(K);
				//std::vector<double> noise = Random::random->randn(K);		// random noise.

				// get the gradients
				// grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
				// WATCH OUT: if the expression is -n / phi_sum, the unsigned n is converted to something
				// huge and positive.
				//std::vector<double> grads(K, -(n / phi_i_sum));		// Hard to grasp... RFHH
				double* grads = new double[K]();
				for (int k = 0; k < K; k++){
					grads[k] = -1 *n / phi_i_sum;
				}

				for (int k = 0; k < K; k++) {
					grads[k] += 1.0 / phi[i][k] * z[k];
				}

				// update the phi
				for (int k = 0; k < K; k++) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
					// FIXME RFHH a**0.5 * b**0.5 better written as sqrt(a*b) ?
					(*phi_star)[i][k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] +
						grads[k]) +
						std::pow(eps_t, .5) * std::pow(phi[i][k], .5) * noise[k]);
#else
					double f = std::sqrt(eps_t * phi[i][k]);
					(*phi_star)[i][k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] +
						grads[k]) +
						f * noise[k]);
#endif
				}
				delete[] grads;
				delete[] noise;
			
			}


			// could leave this function as it is, but we don't use this for now... [wenzhe]
			int sample_z_for_each_edge(double y, const std::vector<double> &pi_a, const std::vector<double> &pi_b, const std::vector<double> &beta, int K) const {
				/**
				* sample latent variables z_ab and z_ba
				* but we don't need to consider all of the cases. i.e  (z_ab = j, z_ba = q) for all j and p.
				* because of the gradient depends on indicator function  I(z_ab=z_ba=k), we only need to consider
				* K+1 different cases:  p(z_ab=0, z_ba=0|*), p(z_ab=1,z_ba=1|*),...p(z_ab=K, z_ba=K|*),.. p(z_ab!=z_ba|*)
				*
				* Arguments:
				*   y:        observation [0,1]
				*   pi_a:     community membership for node a
				*   pi_b:     community membership for node b
				*   beta:     community strengh.
				*   epsilon:  model parameter.
				*   K:        number of communities.
				*
				* Returns the community index. If it falls into the case that z_ab!=z_ba, then return -1
				*/
				std::vector<double> p(K + 1);
#ifdef EFFICIENCY_FOLLOWS_PYTHON
				// FIXME pow for selecting after y = 0 or 1
				for (int k = 0; k < K; k++) {
					p[k] = std::pow(beta[k], y) * pow(1 - beta[k], 1 - y) * pi_a[k] * pi_b[k];
				}
#else
				if (y == 1) {
					for (int k = 0; k < K; k++) {
						p[k] = beta[k] * pi_a[k] * pi_b[k];
					}
				}
				else {
					for (int k = 0; k < K; k++) {
						p[k] = (1 - beta[k]) * pi_a[k] * pi_b[k];
					}
				}
#endif
				// p[K] = 1 - np.sum(p[0:K])
				p[K] = 1.0 - std::accumulate(p.begin(), p.begin() + K, 0.0);

				// sample community based on probability distribution p.
				// bounds = np.cumsum(p)
				std::vector<double> bounds(K + 1);
				std::partial_sum(p.begin(), p.end(), bounds.begin());
				double location = Random::random->random() * bounds[K];

				// get the index of bounds that containing location.
				for (int i = 0; i < K; i++) {
					if (location <= bounds[i]) {
						return i;
					}
				}

				return -1;
			}

			
			// could leave this function as it is, but we don't use this for now...  [wenzhe]
			std::vector<int> sample_latent_vars(int node, const OrderedVertexSet &neighbor_nodes) const {
				/**
				* given a node and its neighbors (either linked or non-linked), return the latent value
				* z_ab for each pair (node, neighbor_nodes[i].
				*/
				std::vector<int> z(K, 0);
				for (auto neighbor = neighbor_nodes.begin();
					neighbor != neighbor_nodes.end();
					neighbor++) {
					int y_ab = 0;      // observation
					Edge edge(std::min(node, *neighbor), std::max(node, *neighbor));
					if (edge.in(network.get_linked_edges())) {
						y_ab = 1;
					}

					int z_ab = 0; // TODO
					//int z_ab = sample_z_ab_from_edge(y_ab, pi[node], pi[*neighbor], beta, epsilon, K);
					assert(z_ab >= 0);
					assert(z_ab < (int)z.size());
					z[z_ab]++;
				}

				return z;
			}


			/**
			* run mini-batch based MCMC sampler.  This sampler is pretty much completed.. may test it first?
			*/
			virtual void run() {
				// pr = cProfile.Profile()
				// pr.enable()
				clock_t t1, t2;
				std::vector<double> timings;
				t1 = clock();

				while (step_count < max_iteration && !is_converged()) {
					
					//print "step: " + str(self._step_count)
					cout<<"calling hold out...";
					double ppx_score = cal_perplexity_held_out();
					std::cout << std::fixed << std::setprecision(12) << "perplexity for hold out set: " << ppx_score << std::endl;
					ppxs_held_out.push_back(ppx_score);

					t2 = clock();
					float diff = ((float)t2 - (float)t1);
					float seconds = diff / CLOCKS_PER_SEC;
					timings.push_back(seconds);

					/*  wenzhe commented this out...
					std::vector<std::vector<double> > phi_star(pi);
					// iterate through each node, and update parameters pi_a
					for (int i = 0; i < N; i++) {
					// update parameter for pi_i
					//print "updating: " + str(i)
					auto neighbor_nodes = sample_neighbor_nodes_batch(i);

					std::vector<int> z = sample_latent_vars(i, neighbor_nodes);
					update_pi_for_node(i, z, &phi_star, neighbor_nodes.size());

					}

					phi = phi_star;
					*/

					// wenzhe's version
					for (int i = 0; i < N; i++){
						auto neighbor_nodes = sample_neighbor_nodes_batch(i);
						update_phi(i, neighbor_nodes);	// update phi for node i
					}
					//np::row_normalize(&pi, phi);	// update pi from phi. 
					// update beta
					update_pi_from_phi();

					update_beta();

					// write into file 
					if (step_count% 1000 == 1){
						ofstream myfile;
  						myfile.open ("mcmc_batch_20_us_air.txt");
  						int size = ppxs_held_out.size();
  						for (int i = 0; i < size; i++){
  							int iteration = i * 1 + 1;
  							myfile <<iteration<<"    "<<timings[i]<<"    "<<ppxs_held_out[i]<<"\n";
  						}
  						
  						myfile.close();
					}


					step_count++;
					

				}

#if 0
				pr.disable()
					s = StringIO.StringIO()
					sortby = 'cumulative'
					ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
					ps.print_stats()
					print s.getvalue()
#endif
			}


			// wenzhe's version. The version is different from previous one that:
			// in the previous version, we use gibbs sampler to sample from distribution
			// but in this version, we actually analytically calculate the probability
			void update_phi(int i, const OrderedVertexSet &neighbor_nodes){
				double eps_t = a * std::pow(1 + step_count / b, -c);	// step size
				double phi_i_sum = getSum(phi[i],K);
				//std::vector<double> grads(K);							// gradient for K classes
				double* grads = new double[K]();
				//std::vector<double> phi_star(K);                        // temp vars
				//std::vector<double> noise = Random::random->randn(K);	// random gaussian noise.
				double* noise = Random::random->randnArray(K);
				for (auto neighbor = neighbor_nodes.begin();
					neighbor != neighbor_nodes.end();
					neighbor++) {

					int y_ab = 0;      // observation
					Edge edge(std::min(i, *neighbor), std::max(i, *neighbor));
					if (edge.in(network.get_linked_edges())) {
						y_ab = 1;
					}

					//std::vector<double> probs(K);
					double* probs = new double[K]();
					for (int k = 0; k < K; k++){
						probs[k] = std::pow(beta[k], y_ab) * std::pow(1 - beta[k], 1 - y_ab) * pi[i][k] * pi[*neighbor][k];
						probs[k] += std::pow(epsilon, y_ab) * std::pow(1 - epsilon, 1 - y_ab) * pi[i][k] * (1 - pi[*neighbor][k]);
					}

					double prob_sum = getSum(probs, K);
					for (int k = 0; k < K; k++){
						grads[k] += (probs[k] / prob_sum) / phi[i][k] - 1.0 / phi_i_sum;
					}
					
					delete[] probs;
				}

				
				// update phi for node i
				for (int k = 0; k < K; k++){
					phi[i][k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] + grads[k]) + std::pow(eps_t, 0.5)*std::pow(phi[i][k], 0.5) *noise[k]);
				}
				delete[] noise;
				delete[] grads;
			
				// assign back to phi. 
				//phi[i] = phi_star;
			}


			/*
			update beta. Instead of sampling, we calculate the probability directly.
			*/
			void update_beta() {
				double** grads;
				grads = new double*[K];
				for (int k = 0; k < K; k++){
					grads[k] = new double[2]();
				}
				//std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));
				//theta_sum = np.sum(self.__theta,1)
				// TODO....
				//std::vector<double> theta_sum(theta.size());
				//std::transform(theta.begin(), theta.end(), theta_sum.begin(), np::sum<double>);
				double* theta_sum = new double[K];
				for (int k = 0; k<K; k++){
					theta_sum[k] = 0;
				}
				for (int k = 0; k < K; k++){
					theta_sum[k] = theta[k][0] + theta[k][1];
				}


				// update gamma, only update node in the grad
				double eps_t = eps_t = a * std::pow(1 + step_count / b, -c);

				// wenzhe commented it out....
				//if (! stepsize_switch) {
				//    eps_t = std::pow(1024+step_count, -0.5);
				//} else {
				//   eps_t = a * std::pow(1 + step_count / b, -c);
				//}

				for (int i = 0; i < N; i++) {
					for (int j = i + 1; j < N; j++) {
						Edge edge(i, j);

						if (edge.in(network.get_held_out_set()) || edge.in(network.get_test_set())) {
							continue;
						}

						int y = 0;
						if (edge.in(network.get_linked_edges())) {
							y = 1;
						}

						// wenzhe's version
						double* probs = new double[K]();
						//std::vector<double> probs(K);
						double pi_sum = 0.0;
						for (int k = 0; k < K; k++){
							pi_sum += pi[i][k] * pi[j][k];
							probs[k] = std::pow(beta[k], y) * std::pow(1 - beta[k], 1 - y) * pi[i][k] * pi[j][k];
						}

						double prob_0 = std::pow(epsilon, y) * std::pow(1 - epsilon, 1 - y) * (1 - pi_sum);
						double prob_sum = getSum(probs,K) + prob_0;
						for (int k = 0; k < K; k++){
							grads[k][0] += (probs[k] / prob_sum) * (std::abs(1 - y) / theta[k][0] - 1 / theta_sum[k]);
							grads[k][1] += (probs[k] / prob_sum) * (std::abs(-y) / theta[k][1] - 1 / theta_sum[k]);
						}

						delete[] probs;

						/*	wenzhe commented it out...
						int z = sample_z_for_each_edge(y_ab, pi[i], pi[j],
						beta, K);
						if (z == -1) {
						continue;
						}

						grads[z][0] += std::abs(1-y_ab) / theta[z][0] - 1/ sums[z];
						grads[z][1] += std::abs(-y_ab) / theta[z][1] - 1 / sums[z];
						*/
					}
				}

				// update theta
				double** noise;
				noise = new double*[K];
				for (int k = 0; k < K; k++){
					noise[k] = Random::random->randnArray(2);
				}
				//std::vector<std::vector<double> > noise = Random::random->randn(K, 2);
				//std::vector<std::vector<double> > theta_star(theta);
				for (int k = 0; k < K; k++) {
					for (int i = 0; i < 2; i++) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
						// FIXME rewrite a**0.5 * b**0.5 as sqrt(a * b)
						theta_star[k][i] = std::abs(theta[k][i] + eps_t / 2 * (eta[i] - theta[k][i] + \
							grads[k][i]) +
							std::pow(eps_t, .5) * pow(theta[k][i], .5) * noise[k][i]);
#else
						double f = std::sqrt(eps_t * theta[k][i]);
						theta[k][i] = std::abs(theta[k][i] + eps_t / 2 * (eta[i] - theta[k][i] + \
							grads[k][i]) +
							f * noise[k][i]);
#endif
					}
				}
				//theta = theta_star;
				for (int k = 0; k < K; k++){
					delete[] noise[k];
				}
				delete[] noise;

				delete[] theta_sum;

				for (int k = 0; k < K; k++){
					delete[] grads[k];
				}
				delete[] grads;
				// wenzhe commented out... don't need...
				//if (step_count < 50000) {
				//	theta = theta_star;
				//} else {
				//	// self.__theta = theta_star * 1.0/(self._step_count) + (1-1.0/(self._step_count))*self.__theta
				//	double inv_step_count = 1.0 / step_count;
				//	double one_inv_step_count = 1.0 - inv_step_count;
				//	MCMCMyOp<double> myOp(inv_step_count, one_inv_step_count);
				//	for (int k = 0; k < theta.size(); k++) {
				//		std::transform(theta[k].begin(), theta[k].end(),
				//					   theta_star[k].begin(),
				//					   theta[k].begin(),
				//					   myOp);
				//	}
				//}

				//self.__theta = theta_star
				// update beta from theta
				// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
				// self._beta = temp[:,1]
				//std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
				//np::row_normalize(&temp, theta);
				//std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
				update_beta_from_theta();
			}


		protected:
			// replicated in both mcmc_sampler_
			double	a;
			double	b;
			double	c;

			int num_node_sample;						// used for sampling neighborhood nodes

			//std::vector<std::vector<double> > theta;		// parameterization for \beta
			//std::vector<std::vector<double> > phi;			// parameterization for \pi
			double** theta;
			double** phi;
		};

	}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_BATCH_H__
