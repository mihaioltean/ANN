#include <stdio.h>


#include "ann.h"
#include "read_data.h"


t_ann ann;

//---------------------------------------------------------------------------

int main()
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

	//ann.to_file("ann2.txt");

	int class_index;

	double *out = new double[ann.get_num_neurons(ann.get_num_layers() - 1)];

	int num_incorrectly_classified = 0;

	for (int i = 0; i < num_data; i++) {
		ann.test(test_data[i], out, class_index);
		printf("%d\n", class_index);

		if (target[i][class_index] != 1) // we have value 1 on that position if it belongs to class_index
			num_incorrectly_classified++;
	}

	printf("num_incorrectly_classified = %d\n", num_incorrectly_classified);

	ann.compute_num_incorrectly_classified(test_data, target, num_data);

	printf("num_incorrectly_classified = %d\n", ann.get_num_incorrectly_classified());

	delete[] out;
	delete_data(test_data, target, num_data);

	ann.release_memory();

	getchar();

	return 0;
}
//------------------------------------------------------------------------