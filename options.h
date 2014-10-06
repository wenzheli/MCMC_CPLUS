#ifndef OPTIONS_H_
#define OPTIONS_H_

class Options{
public:
	double alpha;	
	double eta0;
	double eta1;
	double epsilon;
	int K;
	int max_iteration;
	int mini_batch_size;
	double a, b, c;
	string dataset_class;
	string filename;
};

#endif