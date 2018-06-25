// ann.cpp : Defines the entry point for the console application.
//

#include "ann.h"
#include <stdio.h>

t_ann ann;

//---------------------------------------------------------------------------
void allocate_training_data(double **&data, double **&target, int num_training_data, int num_variables, int num_outputs)
{
	target = new double*[num_training_data];
	data = new double*[num_training_data];
	for (int i = 0; i < num_training_data; i++) {
		data[i] = new double[num_variables];
		target[i] = new double[num_outputs];
	}
}
//---------------------------------------------------------------------------
bool read_file(const char *filename, double **&training_data, double **&target, int &num_training_data, int &num_variables, int &num_outputs)
{
	FILE* f = fopen(filename, "r");
	if (!f)
		return false;

	fscanf(f, "%d%d%d", &num_training_data, &num_variables, &num_outputs);
	allocate_training_data(training_data, target, num_training_data, num_variables, num_outputs);

	for (int i = 0; i < num_training_data; i++) {
		for (int j = 0; j < num_variables; j++)
			fscanf(f, "%lf", &training_data[i][j]);

		// read one output
		int class_index;
		fscanf(f, "%d", &class_index);
		for (int j = 0; j < num_outputs; j++)
			target[i][j] = 0;
		target[i][class_index] = 1;
	}
	fclose(f);
	return true;
}
//---------------------------------------------------------------------------
void delete_data(double **&data, double **&target, int num_training_data)
{
	if (data) {
		for (int i = 0; i < num_training_data; i++)
			delete[] data[i];

		delete[] data;
	}
	if (target) {
		for (int i = 0; i < num_training_data; i++)
			delete[] target[i];

		delete[] target;
	}
}
//---------------------------------------------------------------------------
void f(void)
{
	printf("epoch = %d error = %lf\n", ann.get_epoch(), ann.get_error());
}
//------------------------------------------------------------------------
int main()
{
	double **training_data;
	double **target;
	int num_data;

	int num_variables;
	int num_outputs;

	printf("reading data ... ");
	if (!read_file("c:/Mihai/uab/ann/data/mnist_test.txt", training_data, target, num_data, num_variables, num_outputs)) {
		printf("Cannot read file!\n");
		getchar();
		return 1;
	}
	
	printf("done\n");
		
		ann.set_num_layers(3);
		ann.set_num_neurons(0, num_variables);
		ann.set_num_neurons(1, 4);
		ann.set_num_neurons(2, num_outputs);

		ann.set_learning_rate(0.01);
		ann.set_num_iterations(10);

		ann.train(training_data, target, num_data, f);
		
		double error = ann.get_error();
	
		
	printf("Error = %lf\n", error);

	delete_data(training_data, target, num_data);

	ann.release_memory();

	getchar();

    return 0;
}
//------------------------------------------------------------------------