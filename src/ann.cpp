
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
}
//------------------------------------------------------
t_ann::~t_ann()
{
	release_memory();
}
//------------------------------------------------------
void t_ann::allocate_memory(void)
{
	// allocate weights
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
}
//------------------------------------------------------
void t_ann::release_memory(void)
{
// delete out
	for (int i = 0; i < num_layers; i++)
		delete[] out[i];
	delete[] out;

	// delete weights
	for (int layer = 0; layer < num_layers - 1; layer++) {
		for (int neuron = 0; neuron < num_neurons[layer + 1]; neuron++)
			delete[] weights[layer][neuron];
		delete[] weights[layer];
	}
	delete[] weights;

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
void t_ann::train(double ** training_data, double **target, int num_data)
{
	allocate_memory();
	init_weights();

	compute_error(training_data, target, num_data);

	for (int epoch = 0; epoch < num_iterations; epoch++) {
		// printf eroare

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
			for (int n2 = 0; n2 < num_neurons[num_layers - 1]; n2++)
				for (int n1 = 0; n1 < num_neurons[num_layers - 1] + 1; n1++)
					weights[num_layers - 2][n2][n1] -= learning_rate * (out[num_layers - 1][n2] - target[data_index][n2]) * out[num_layers - 1][n2] * (1 - out[num_layers - 1][n2]) * out[num_layers - 2][n1];

	}
}
//------------------------------------------------------