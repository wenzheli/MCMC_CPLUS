#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "np.h"
#include "myrandom.h"
// #include "mcmc/sample_latent_vars.h"

#include "learner.h"
#include "mcmc_sampler_batch.h"

namespace mcmc {
	namespace learning {

		// typedef std::unordered_map<Edge, int>	EdgeMapZ;
		typedef std::map<Edge, int>	EdgeMapZ;

		class MCMCSamplerStochastic : public Learner {
		public:
			/**
			Mini-batch based MCMC sampler for community overlapping problems. Basically, given a
			connected graph where each node connects to other nodes, we try to find out the
			community information for each node.

			Formally, each node can be belong to multiple communities which we can represent it by
			distribution of communities. For instance, if we assume there are total K communities
			in the graph, then each node a, is attached to community distribution \pi_{a}, where
			\pi{a} is K dimensional vector, and \pi_{ai} represents the probability that node a
			belongs to community i, and \pi_{a0} + \pi_{a1} +... +\pi_{aK} = 1

			Also, there is another parameters called \beta representing the community strength, where
			\beta_{k} is scalar.

			In summary, the model has the parameters:
			Prior: \alpha, \eta
			Parameters: \pi, \beta
			Latent variables: z_ab, z_ba
			Observations: y_ab for every link.

			And our goal is to estimate the posterior given observations and priors:
			p(\pi,\beta | \alpha,\eta, y).

			Because of the intractability, we use MCMC(unbiased) to do approximate inference. But
			different from classical MCMC approach, where we use ALL the examples to update the
			parameters for each iteration, here we only use mini-batch (subset) of the examples.
			This method is great marriage between MCMC and stochastic methods.
			*/
			MCMCSamplerStochastic(const Options &args, const Network &graph)
				: Learner(args, graph) {

				// step size parameters.
				this->a = args.a;
				this->b = args.b;
				this->c = args.c;

				// control parameters for learning
				 //num_node_sample = static_cast< ::size_t>(std::sqrt(network.get_num_nodes()));
				// TODO: automative update.....
				num_node_sample = N/10;

				// model parameters and re-parameterization
				// since the model parameter - \pi and \beta should stay in the simplex,
				// we need to restrict the sum of probability equals to 1.  The way we
				// restrict this is using re-reparameterization techniques, where we
				// introduce another set of variables, and update them first followed by
				// updating \pi and \beta.
				std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" << std::endl;
				// theta = Random::random->gamma(eta[0], eta[1], K, 2);		// parameterization for \beta
				//theta = Random::random->gamma(100.0, 0.01, K, 2);		// parameterization for \beta
				theta = Random::random->gamma(1, 1, K, 2);
				phi = Random::random->gamma(1, 1, N, K);					// parameterization for \pi

				// FIXME RFHH -- code sharing with variational_inf*::update_pi_beta()
				// temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
				// self._beta = temp[:,1]
				std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
				np::row_normalize(&temp, theta);
				std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
				// self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
				pi.resize(phi.size(), std::vector<double>(phi[0].size()));
				np::row_normalize(&pi, phi);
			}

			virtual ~MCMCSamplerStochastic() {
			}

			virtual void run() {
				/** run mini-batch based MCMC sampler, based on the sungjin's note */
				
				while (step_count < max_iteration && !is_converged()) {
					auto l1 = std::chrono::system_clock::now();

					if (step_count % 10 == 0){

						double ppx_score = cal_perplexity_held_out();
						std::cout << std::fixed << std::setprecision(12) << "step count: "<<step_count<<"perplexity for hold out set: " << ppx_score << std::endl;
						ppxs_held_out.push_back(ppx_score);
					}
					//print "step: " + str(self._step_count)
					/**
					pr = cProfile.Profile()
					pr.enable()
					*/

					// (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
					EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
					const OrderedEdgeSet &mini_batch = *edgeSample.first;
					double scale = edgeSample.second;

					//std::unordered_map<int, std::vector<int> > latent_vars;
					//std::unordered_map<int, ::size_t> size;

					// iterate through each node in the mini batch.
					OrderedVertexSet nodes = nodes_in_batch(mini_batch);
					for (auto node = nodes.begin();
						node != nodes.end();
						node++){
						//cout<<"current node is: "<<*node<<endl;
						OrderedVertexSet neighbors = sample_neighbor_nodes(num_node_sample, *node);
						update_phi(*node, neighbors);
					}
					np::row_normalize(&pi, phi);	// update pi from phi. 
					// update beta
					update_beta(mini_batch, scale);

					delete edgeSample.first;

					step_count++;
					auto l2 = std::chrono::system_clock::now();
					//std::cout << "LOOP  = " << (l2 - l1).count() << std::endl;
				}
			}


		protected:
			void update_beta(const OrderedEdgeSet &mini_batch, double scale){
				std::vector<std::vector<double> > grads(K, std::vector<double>(2, 0.0));
				//theta_sum = np.sum(self.__theta,1)
				// TODO....
				std::vector<double> theta_sum(theta.size());
				std::transform(theta.begin(), theta.end(), theta_sum.begin(), np::sum<double>);

				// update gamma, only update node in the grad
				double eps_t = eps_t = a * std::pow(1 + step_count / b, -c);
				for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++){
					
					int y = 0;
					if (edge->in(network.get_linked_edges())) {
						y = 1;
					}
					int i = edge->first;
					int j = edge->second;

					std::vector<double> probs(K);
					double pi_sum = 0.0;
					for (int k = 0; k < K; k++){
						pi_sum += pi[i][k] * pi[j][k];
						probs[k] = std::pow(beta[k], y) * std::pow(1 - beta[k], 1 - y) * pi[i][k] * pi[j][k];
					}

					double prob_0 = std::pow(epsilon, y) * std::pow(1 - epsilon, 1 - y) * (1 - pi_sum);
					double prob_sum = np::sum(probs) + prob_0;
					for (int k = 0; k < K; k++){
						grads[k][0] += (probs[k] / prob_sum) * (std::abs(1 - y) / theta[k][0] - 1 / theta_sum[k]);
						grads[k][1] += (probs[k] / prob_sum) * (std::abs(-y) / theta[k][1] - 1 / theta_sum[k]);
					}
				}

				// update theta
				std::vector<std::vector<double> > noise = Random::random->randn(K, 2);
				std::vector<std::vector<double> > theta_star(theta);
				for (::size_t k = 0; k < K; k++) {
					for (::size_t i = 0; i < 2; i++) {

						double f = std::sqrt(eps_t * theta[k][i]);
						theta_star[k][i] = std::abs(theta[k][i] + eps_t / 2 * (eta[i] - theta[k][i] + \
							scale * grads[k][i]) +
							f * noise[k][i]);
					}
				}
				theta = theta_star;

				std::vector<std::vector<double> > temp(theta.size(), std::vector<double>(theta[0].size()));
				np::row_normalize(&temp, theta);
				std::transform(temp.begin(), temp.end(), beta.begin(), np::SelectColumn<double>(1));
			}
			

			void update_phi(int i, const OrderedVertexSet &neighbors){
				double eps_t = a * std::pow(1 + step_count / b, -c);	// step size
				double phi_i_sum = np::sum(phi[i]);
				std::vector<double> grads(K);							// gradient for K classes
				std::vector<double> phi_star(K);                        // temp vars
				std::vector<double> noise = Random::random->randn(K);	// random gaussian noise.

				for (auto neighbor = neighbors.begin();
					neighbor != neighbors.end();
					neighbor++) {

					if (i == *neighbor){
						continue;
					}

					int y_ab = 0;      // observation
					Edge edge(std::min(i, *neighbor), std::max(i, *neighbor));
					if (edge.in(network.get_linked_edges())) {
						y_ab = 1;
					}

					std::vector<double> probs(K);
					for (int k = 0; k < K; k++){
						probs[k] = std::pow(beta[k], y_ab) * std::pow(1 - beta[k], 1 - y_ab) * pi[i][k] * pi[*neighbor][k];
						probs[k] += std::pow(epsilon, y_ab) * std::pow(1 - epsilon, 1 - y_ab) * pi[i][k] * (1 - pi[*neighbor][k]);
					}

					double prob_sum = np::sum(probs);
					for (int k = 0; k < K; k++){
						grads[k] += (probs[k] / prob_sum) / phi[i][k] - 1.0 / phi_i_sum;
					}
				}
				// update phi for node i
				for (int k = 0; k < K; k++){
					phi_star[k] = std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] + (N*1.0 / num_node_sample) *grads[k]) + std::pow(eps_t, 0.5)*std::pow(phi[i][k], 0.5) *noise[k]);
				}

				// assign back to phi. 
				phi[i] = phi_star;
			}

			// TODO FIXME make VertexSet an out parameter
			OrderedVertexSet sample_neighbor_nodes(::size_t sample_size, int nodeId) {
				/**
				Sample subset of neighborhood nodes.
				*/
				int p = (int)sample_size;
				OrderedVertexSet neighbor_nodes;
				const EdgeMap &held_out_set = network.get_held_out_set();
				const EdgeMap &test_set = network.get_test_set();

				while (p > 0) {
#ifdef EFFICIENCY_FOLLOWS_PYTHON
					std::cerr << "FIXME: horribly inefficient xrange thingy" << std::endl;
					auto nodeList = Random::random->sample(np::xrange(0, N), sample_size * 2);
#else
					auto nodeList = Random::random->sampleRange(N, sample_size * 2);
#endif
					for (std::vector<int>::const_iterator neighborId = nodeList->begin();
						neighborId != nodeList->end();
						neighborId++) {
						if (p < 0) {
							if (p != 0) {
								//						std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
							}
							break;
						}
						if (*neighborId == nodeId) {
							continue;
						}
						// check condition, and insert into mini_batch_set if it is valid.
						Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
						if (edge.in(held_out_set) || edge.in(test_set) || neighbor_nodes.find(*neighborId) != neighbor_nodes.end()) {
							continue;
						}
						else {
							// add it into mini_batch_set
							neighbor_nodes.insert(*neighborId);
							p -= 1;
						}
					}

					delete nodeList;
				}

				return neighbor_nodes;
			}

			OrderedVertexSet nodes_in_batch(const OrderedEdgeSet &mini_batch) const {
				/**
				Get all the unique nodes in the mini_batch.
				*/
				OrderedVertexSet node_set;
				for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
					node_set.insert(edge->first);
					node_set.insert(edge->second);
				}

				return node_set;
			}

#if 0
			def _save(self) :
				f = open('ppx_mcmc.txt', 'wb')
				for i in range(0, len(self._avg_log)) :
					f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) + "\n")
					f.close()
#endif

#if 0

			int sample_z_ab_from_edge(int y,
					const std::vector<double> &pi_a,
					const std::vector<double> &pi_b,
					const std::vector<double> &beta,
					double epsilon, ::size_t K, int node, int neighbor) const {
				std::vector<double> p(K);

#ifdef EFFICIENCY_FOLLOWS_PYTHON
				for (::size_t i = 0; i < K; i++) {
					// FIMXE lift common expressions
					double tmp = std::pow(beta[i], y) * std::pow(1 - beta[i], 1 - y) * pi_a[i] * pi_b[i];
					// tmp += std::pow(epsilon, y) * std::pow(1-epsilon, 1-y) * pi_a[i] * (1 - pi_b[i]);
					double fac = std::pow(epsilon, y) * std::pow(1.0 - epsilon, 1 - y);
					tmp += fac * pi_a[i] * (1 - pi_b[i]);
					p[i] = tmp;
				}
#else
				if (y == 1) {
					for (::size_t i = 0; i < K; i++) {
						// p[i] = beta[i] * pi_a[i] * pi_b[i] + epsilon * pi_a[i] * (1 - pi_b[i])
						//      = pi_a[i] * (beta[i] * pi_b[i] + epsilon * (1 - pi_b[i]))
						//      = pi_a[i] * (pi_b[i] * (beta[i] - epsilon) + epsilon)
						p[i] = pi_a[i] * (pi_b[i] * (beta[i] - epsilon) + epsilon);
					}
				}
				else {
					double one_eps = 1.0 - epsilon;
					for (::size_t i = 0; i < K; i++) {
						// p[i] = (1 - beta[i]) * pi_a[i] * pi_b[i] + (1 - epsilon) * pi_a[i] * (1 - pi_b[i])
						//      = pi_a[i] * ((1 - beta[i]) * pi_b[i] + (1 - epsilon) * (1 - pi_b[i]))
						//      = pi_a[i] * (pi_b[i] * (1 - beta[i] - (1 - epsilon)) + (1 - epsilon) * 1)
						//      = pi_a[i] * (pi_b[i] * (-beta[i] + epsilon) + 1 - epsilon)
						p[i] = pi_a[i] * (pi_b[i] * (epsilon - beta[i]) + one_eps);
					}
				}
#endif

				for (::size_t k = 1; k < K; k++) {
					p[k] += p[k - 1];
				}

				double r = Random::random->random();
				double location = r * p[K - 1];
				// get the index of bounds that containing location.
				for (::size_t i = 0; i < K; i++) {
					if (location <= p[i]) {
						return i;
					}
				}

				// failed, should not happen!
				return -1;
				}

#endif 

		void setNumNodeSample(int numSample){
			num_node_sample = numSample;
		}

		protected:
			// replicated in both mcmc_sampler_
			double	a;
			double	b;
			double	c;

			::size_t num_node_sample;

			//std::vector<std::vector<double> > theta;		// parameterization for \beta
			//std::vector<std::vector<double> > phi;			// parameterization for \pi
			double** theta;
			double** phi;
		};

	}	// namespace learning
}	// namespace mcmc



#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
