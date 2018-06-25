
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "ann.h"

//------------------------------------------------------
t_ann::t_ann()
{
	num_layers = 0;
	num_neurons = NULL;  // num neurons on each layer
	weights = NULL; // weights[0][0][0] = weight of connection between the first neuron on the first layer
					   // and the first neuron on the second layer
	out = NULL;

	num_iterations = 10000;
	deltas = NULL;
	epoch = 0;
}
//------------------------------------------------------
t_ann::~t_ann()
{
	release_memory();
}
//------------------------------------------------------
void t_ann::allocate_memory(void)
{
	// allocate out
	out = new double *[num_layers];
	for (int i = 0; i < num_layers - 1; i++)
		out[i] = new double[num_neurons[i] + 1];
	out[num_layers - 1] = new double[num_neurons[num_layers - 1]];

	// allocate weights
	weights = new double**[num_layers - 1];
	for (int layer = 0; layer < num_layers - 1; layer++) {
		weights[layer] = new double*[num_neurons[layer + 1]];
		for (int neuron = 0; neuron < num_neurons[layer + 1]; neuron++)
			weights[layer][neuron] = new double[num_neurons[layer] + 1];
	}

	// allocate deltas
	deltas = new double*[num_layers - 1];
	for (int layer = 0; layer < num_layers - 1; layer++)
		deltas[layer] = new double[num_neurons[layer + 1]];
}
//------------------------------------------------------
void t_ann::release_memory(void)
{
// delete out
	if (out) {
		for (int i = 0; i < num_layers; i++)
			delete[] out[i];
		delete[] out;
		out = NULL;
	}

	// delete weights
	if (weights) {
		for (int layer = 0; layer < num_layers - 1; layer++) {
			for (int neuron = 0; neuron < num_neurons[layer + 1]; neuron++)
				delete[] weights[layer][neuron];
			delete[] weights[layer];
		}
		delete[] weights;
		weights = NULL;
	}

	// deltas
	if (deltas) {
		for (int i = 0; i < num_layers - 1; i++)
			delete[] deltas[i];
		delete[] deltas;
		deltas = NULL;
	}

	if (num_neurons) {
		delete[] num_neurons;
		num_neurons = NULL;
	}

}
//------------------------------------------------------
void t_ann::set_num_layers(int num_layers)
{
	this->num_layers = num_layers;
	if (num_layers > 0)
		num_neurons = new int[num_layers];
}
//------------------------------------------------------
void t_ann::set_num_neurons(int layer_index, int num_neurons)
{
	this->num_neurons[layer_index] = num_neurons;
}
//------------------------------------------------------
void t_ann::set_num_iterations(int num_iterations)
{
	this->num_iterations = num_iterations;
}
//------------------------------------------------------
int t_ann::get_num_layers(void)
{
	return num_layers;
}
//------------------------------------------------------
int t_ann::get_num_neurons(int layer_index)
{
	return num_neurons[layer_index];
}
//------------------------------------------------------
double t_ann::get_weight(int layer_index, int neuron_index, int weight_index)
{
	return 0; /// must be implemented
}
//------------------------------------------------------
int t_ann::get_num_iterations(void)
{
	return num_iterations;
}
//------------------------------------------------------
void t_ann::init_weights(void)
{
	for (int layer = 0; layer < num_layers - 1; layer++) {
		for (int neuron = 0; neuron < num_neurons[layer + 1]; neuron++)
			for (int w = 0; w < num_neurons[layer] + 1; w++)
				weights[layer][neuron][w] = rand() / (double)RAND_MAX - 0.5;
	}
}
//------------------------------------------------------
int t_ann::get_epoch(void)
{
	return epoch;
}
//-------------------------------------------------------
double t_ann::get_error(void)
{
	return ann_error;
}
//------------------------------------------------------
double logistic_function(double x)
{
	return 1 / (1 + exp(-x));
}
//------------------------------------------------------
void t_ann::set_learning_rate(double new_learning_rate)
{
	learning_rate = new_learning_rate;
}
//------------------------------------------------------
void t_ann::compute_error(double **training_data, double **target, int num_data)
{
	ann_error = 0;

	for (int data_index = 0; data_index < num_data; data_index++) {
		// set input data
		for (int input = 0; input < num_neurons[0]; input++)
			out[0][input] = training_data[data_index][input];
		// compute out for each other layer
		for (int layer = 1; layer < num_layers; layer++) {
			for (int n2 = 0; n2 < num_neurons[layer]; n2++) {
				out[layer][n2] = 0;
				for (int w = 0; w < num_neurons[layer - 1] + 1; w++)
					out[layer][n2] += weights[layer - 1][n2][w] * out[layer - 1][w];
				out[layer][n2] = logistic_function(out[layer][n2]);
			}
		}
		for (int neuron = 0; neuron < num_neurons[num_layers - 1]; neuron++)
			ann_error += 1 / 2.0 * (out[num_layers - 1][neuron] - target[data_index][neuron])*(out[num_layers - 1][neuron] - target[data_index][neuron]);
	}
}
//------------------------------------------------------
void t_ann::test(double * test_data, double *out_last_layer, int &class_index)
{
	// set input data
	for (int input = 0; input < num_neurons[0]; input++)
		out[0][input] = test_data[input];
	// compute out for each other layer
	for (int layer = 1; layer < num_layers; layer++) {
		for (int n2 = 0; n2 < num_neurons[layer]; n2++) {
			out[layer][n2] = 0;
			for (int w = 0; w < num_neurons[layer - 1] + 1; w++)
				out[layer][n2] += weights[layer - 1][n2][w] * out[layer - 1][w];
			out[layer][n2] = logistic_function(out[layer][n2]);
		}
	}
	int max_out_value = -1;
	class_index = -1;
	for (int neuron = 0; neuron < num_neurons[num_layers - 1]; neuron++) {
		out_last_layer[neuron] = out[num_layers - 1][neuron];
		if (max_out_value < out_last_layer[neuron]) {
			class_index = neuron;
			max_out_value = out_last_layer[neuron];
		}
	}
}
//------------------------------------------------------
bool t_ann::to_file(const char* filename)
{
    FILE *f = fopen(filename, "w");
    if (!f)
        return false;
    
	fprintf(f, "%d\n", num_layers);

	for (int layer = 0; layer < num_layers; layer++)
		fprintf(f, "%d ", num_neurons[layer]);
	fprintf(f, "\n");

	for (int layer = 1; layer < num_layers; layer++)
		for (int n2 = 0; n2 < num_neurons[layer]; n2++)
			for (int w = 0; w < num_neurons[layer - 1] + 1; w++)
				fprintf(f, "%lf ", weights[layer - 1][n2][w]);

    fclose(f);
	return true;
}
//------------------------------------------------------
bool t_ann::from_file(const char* filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return false;
    
	release_memory();

	fscanf(f, "%d", &num_layers);
	if (num_layers <= 0)
		return false;

	num_neurons = new int[num_layers];
	for (int layer = 0; layer < num_layers; layer++)
		fscanf(f, "%d", &num_neurons[layer]);

	// allocate memory
	weights = new double**[num_layers - 1];
	for (int layer = 0; layer < num_layers - 1; layer++) {
		weights[layer] = new double*[num_neurons[layer + 1]];
		for (int neuron = 0; neuron < num_neurons[layer + 1]; neuron++)
			weights[layer][neuron] = new double[num_neurons[layer] + 1];
	}

	// read weights
	for (int layer = 1; layer < num_layers; layer++)
		for (int n2 = 0; n2 < num_neurons[layer]; n2++)
			for (int w = 0; w < num_neurons[layer - 1] + 1; w++)
				fscanf(f, "%lf", &weights[layer - 1][n2][w]);

    fclose(f);
    return true;
}
//------------------------------------------------------
void t_ann::train(double ** training_data, double **target, int num_data, t_func f)
{
	allocate_memory();
	init_weights();

	// first set the value of bias nodes to 1

	for (int layer = 0; layer < num_layers - 1; layer++)
		out[layer][num_neurons[layer]] = 1;

	for (epoch = 0; epoch < num_iterations; epoch++) {
		compute_error(training_data, target, num_data);
		f();

		for (int data_index = 0; data_index < num_data; data_index++) {
			// forward pass
			// set input data
			for (int input = 0; input < num_neurons[0]; input++)
				out[0][input] = training_data[data_index][input];
			// compute out for each other layer
			for (int layer = 1; layer < num_layers; layer++) {
				for (int n2 = 0; n2 < num_neurons[layer]; n2++) {
					out[layer][n2] = 0;
					for (int w = 0; w < num_neurons[layer - 1] + 1; w++)
						out[layer][n2] += weights[layer - 1][n2][w] * out[layer - 1][w];
					out[layer][n2] = logistic_function(out[layer][n2]);
				}
			}
			// backward pass
			// update weights between last layer and hidden layer
			for (int n2 = 0; n2 < num_neurons[num_layers - 1]; n2++) { // last layer
				deltas[num_layers - 2][n2] = (out[num_layers - 1][n2] - target[data_index][n2]) * out[num_layers - 1][n2] * (1 - out[num_layers - 1][n2]);
				for (int neuron_hidden_layer = 0; neuron_hidden_layer < num_neurons[num_layers - 2] + 1; neuron_hidden_layer++) // hidden layer
					weights[num_layers - 2][n2][neuron_hidden_layer] -= learning_rate * deltas[num_layers - 2][n2] * out[num_layers - 2][neuron_hidden_layer];
			}

			// weights between input layer and hidden layer

			for (int second_layer_index = num_layers - 2; second_layer_index > 0; second_layer_index--) {
				for (int neuron_2nd_layer = 0; neuron_2nd_layer < num_neurons[second_layer_index]; neuron_2nd_layer++) {
					deltas[second_layer_index - 1][neuron_2nd_layer] = 0;
					double cst = out[second_layer_index][neuron_2nd_layer] * (1 - out[second_layer_index][neuron_2nd_layer]);
					for (int neuron_3rd_layer = 0; neuron_3rd_layer < num_neurons[second_layer_index + 1]; neuron_3rd_layer++)
						deltas[second_layer_index - 1][neuron_2nd_layer] += deltas[second_layer_index][neuron_3rd_layer] * weights[second_layer_index][neuron_3rd_layer][neuron_2nd_layer] * cst;
					for (int neuron_1st_layer = 0; neuron_1st_layer < num_neurons[second_layer_index - 1]; neuron_1st_layer++)
						weights[second_layer_index - 1][neuron_2nd_layer][neuron_1st_layer] -= learning_rate * deltas[second_layer_index - 1][neuron_2nd_layer] * out[second_layer_index - 1][neuron_1st_layer];
				}
			}

		}
	}
}
//------------------------------------------------------
