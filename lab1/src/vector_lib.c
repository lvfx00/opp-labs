#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double scalar_vector_x_vector(double *left, double *right, int size)
{
    double result = 0;
    int i;
    for (i = 0; i < size; i++)
    {
        result += left[i] * right[i];
    }
    return result;
}

void vector_sub_vector(double *left, double *right, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        left[i] -= right[i];
    }
}

double vector_norm(double *vector, int size)
{
    double result = 0;
    int i;
    for (i = 0; i < size; i++)
    {
        result += pow(vector[i], 2);
    }
    return sqrt(result);
}

void vector_x_scalar(double *vector, double scalar, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        vector[i] *= scalar;
    }
}

void print_double_vector(double *vector, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void print_int_vector(int *vector, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

double *init_double_vector(int size)
{
    double *vect = calloc(size, sizeof(double));
    if(!vect) exit(1);
    return vect;
}

int *init_int_vector(int size)
{
    int *vect = calloc(size, sizeof(int));
    if(!vect) exit(1);
    return vect;
}
