#ifndef ANN_H
#define ANN_H

class t_ann {
private:
	int num_layers; 
	int *num_neurons;  // num neurons on each layer
	double ***weights; // weights[0][0][0] = weight of connection between the first neuron on the first layer
						// and the first neuron on the second layer
	double **out;

	int num_iterations;

public:

	void set_num_layers(int num_layers);
	void set_num_neurons(int layer_index, int num_neurons);
	void set_num_iterations(int num_iterations);

	int get_num_layers(void);
	int get_num_neurons(int layer_index);
	double get_weight(int layer_index, int neuron_index, int weight_index);
	int get_num_iterations(void);

	void train(double ** training_data, double **target, int num_data);
};

#endif
