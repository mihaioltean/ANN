#pragma once

void allocate_training_data(double **&data, double **&target, int num_training_data, int num_variables, int num_outputs);
bool read_file(const char *filename, double **&training_data, double **&target, int &num_training_data, int &num_variables, int &num_outputs);
void delete_data(double **&data, double **&target, int num_training_data);