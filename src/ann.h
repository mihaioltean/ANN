#ifndef ANN_H
#define ANN_H

typedef void(*t_func)(void);


class t_ann {
private:
	int num_layers; 
	int *num_neurons;  // num neurons on each layer
	double ***weights; // weights[0][0][0] = weight of connection between the first neuron on the first layer
						// and the first neuron on the second layer

	double **deltas;
	double **out;

	int num_iterations;

	double ann_error;
	int num_incorrectly_classified;

	double learning_rate;

	int epoch;

public:

	t_ann();
	~t_ann();
	void set_num_layers(int num_layers);
	void set_num_neurons(int layer_index, int num_neurons);
	void set_num_iterations(int num_iterations);
	void set_learning_rate(double new_learning_rate);

	int get_num_layers(void);
	int get_num_neurons(int layer_index);
	double get_weight(int layer_index, int neuron_index, int weight_index);
	int get_num_iterations(void);

	void train(double ** training_data, double **target, int num_data, t_func);
	void test(double * test_data, double *out_last_layer, int &class_index);

	void allocate_memory(void);
	void release_memory(void);
	void init_weights(void);

	double get_error(void);
	int get_num_incorrectly_classified(void);

	void compute_error(double **training_data, double **target, int num_data);
	void compute_num_incorrectly_classified(double **training_data, double **target, int num_data);

	int get_epoch(void);
    
    bool from_file(const char* filename);
    bool to_file(const char* filename);
};

#endif
