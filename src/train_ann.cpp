// ann.cpp : Defines the entry point for the console application.
//

#include "ann.h"
#include <stdio.h>
#include "read_data.h"

t_ann ann;

//---------------------------------------------------------------------------
void f(void)
{
	//printf("epoch = %d error = %lf\n", ann.get_epoch(), ann.get_error());
	printf("epoch = %d error = %lf num_incorrect_class = %d\n", ann.get_epoch(), ann.get_error(), ann.get_num_incorrectly_classified());
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
		ann.set_num_iterations(1000);

		ann.train(training_data, target, num_data, f);
		
		double error = ann.get_error();
		ann.to_file("ann.txt");
		
	printf("Error = %lf\n", error);

	delete_data(training_data, target, num_data);

	ann.release_memory();

	getchar();

    return 0;
}
//------------------------------------------------------------------------