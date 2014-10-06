#include "myrandom.h"


namespace mcmc {
	namespace Random {


#ifdef RANDOM_FOLLOWS_PYTHON
		FileReaderRandom *random = new FileReaderRandom(0);
#else
		Random *random = new Random(0);
#endif

	}	// namespace Random
}	// namespace mcmc
