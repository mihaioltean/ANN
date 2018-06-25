#include <stdio.h>


#include "ann.h"
#include "read_data.h"


t_ann ann;

//---------------------------------------------------------------------------

int _main()
{
	double **test_data;
	double **target;
	int num_data;

	int num_variables;
	int num_outputs;

	printf("reading data ... ");
	if (!read_file("c:/Mihai/uab/ann/data/mnist_test.txt", test_data, target, num_data, num_variables, num_outputs)) {
		printf("Cannot read file!\n");
		getchar();
		return 1;
	}

	printf("done\n");

	if (!ann.from_file("ann.txt")) {
		printf("Cannot read ANN!\n");
		getchar();
		return 2;
	}

	int class_index;

	double *out = new double[ann.get_num_neurons(ann.get_num_layers() - 1)];

	for (int i = 0; i < num_data; i++) {
		ann.test(test_data[i], out, class_index);
		printf("%d\n", class_index);
	}


	delete[] out;
	delete_data(test_data, target, num_data);

	ann.release_memory();

	getchar();

	return 0;
}
//------------------------------------------------------------------------