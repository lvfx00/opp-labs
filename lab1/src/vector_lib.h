#pragma once

//vector operations

double scalar_vector_x_vector(double *left, double *right, int size);

void vector_sub_vector(double* left, double* right, int size);

void vector_x_scalar(double *vector, double scalar, int size);

double vector_norm(double *vector, int size);

void print_double_vector(double *vector, int size);

double *init_double_vector(int size);

void print_int_vector(int *vector, int size);

int *init_int_vector(int size);
