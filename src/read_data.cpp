
#include <stdio.h>

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
