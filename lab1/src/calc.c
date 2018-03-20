#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "calc.h"
#include "vector_lib.h"

#define EPSILON 0.001
#define TAU 0.001

void init(double **matrix, int **lines_num, int **offsets, int N, int comm_rank, int comm_size)
{
    int i; // iterator

    // nummber of lines that is processing for each process
    *lines_num = init_int_vector(comm_size);
    for (i = 0; i < comm_size; ++i) {
        (*lines_num)[i] = N / comm_size;
    }
    // add 1 to first N % comm_size processes
    for (i = 0; i < N % comm_size; ++i) {
        (*lines_num)[i] += 1;
    }

    // number of lines skipped from matrix begining for each process
    *offsets = init_int_vector(comm_size);
    (*offsets)[0] = 0;
    for (i = 1; i < comm_size; ++i) {
        (*offsets)[i] = (*lines_num)[i - 1] + (*offsets)[i - 1];
    }

    // init only lines for this process
    *matrix = init_double_vector((*lines_num)[comm_rank] * N);
    for (i = 0; i < (*lines_num)[comm_rank] * N; i++)
    {
        // 2 for main diagonal elements
        (*matrix)[i] = (i % N == (*offsets)[comm_rank] + i / N) ? 2 : 1;
    }
}

void calc_easy(int comm_size, int comm_rank, int N)
{
    int i;

    int *offsets, *lines_num;
    double *matrix_part;

    init(&matrix_part, &lines_num, &offsets, N, comm_rank, comm_size);

    // initial x value
    double *x_vect = init_double_vector(N); // init and set to zero

    // b vector
    double *b_vect = init_double_vector(N);
    // init b vector
    for (i = 0; i < N; ++i) {
        b_vect[i] = N + 1;
    }

    double b_norm = vector_norm(b_vect, N); // set zero by default

    // Ax - b norm for all processes and current
    double Ax_b_norm;
    double Ax_b_norm_part;
    
    // buffer for entire new Ax-nth - b vector
    double *buffer = init_double_vector(N);

    // buffer for new Ax-nth - b vector for this process
    double *buffer_part = init_double_vector(lines_num[comm_rank]);

    while (1) {
        Ax_b_norm_part = 0;

        for (i = 0; i < lines_num[comm_rank]; i++) {
            // A * x
            buffer_part[i] = scalar_vector_x_vector(matrix_part + i * N, x_vect, N);
            // -b
            buffer_part[i] -= b_vect[offsets[comm_rank] + i];

            Ax_b_norm_part += buffer_part[i] * buffer_part[i];
        }
        
        // collect all buffer_part from all processes to buffer
        MPI_Allgatherv(buffer_part, lines_num[comm_rank], 
            MPI_DOUBLE, buffer, lines_num, offsets, MPI_DOUBLE, MPI_COMM_WORLD);

        // collect all norm parts from processes to one variable
        MPI_Allreduce(&Ax_b_norm_part, &Ax_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // don't forget to take a square root
        Ax_b_norm = sqrt(Ax_b_norm);

        if (Ax_b_norm / b_norm < EPSILON) {
            break;
        }

        vector_x_scalar(buffer, TAU, N);
        vector_sub_vector(x_vect, buffer, N); // now x_vect contains new x-n+1-th value
    }

    if (comm_rank == 0) {
        print_double_vector(x_vect, N);
    }

    free(lines_num);
    free(offsets);
    free(matrix_part);
    free(x_vect);
    free(b_vect);
    free(buffer_part);
    free(buffer);
}

void calc_hard(int comm_size, int comm_rank, int N)
{
    int i;

    int *offsets, *lines_num;
    double *matrix_part;

    init(&matrix_part, &lines_num, &offsets, N, comm_rank, comm_size);

    // initial x value
    // every process has max size of part (because x_part will be shared)
    double *x_vect_part = init_double_vector(N / comm_size + 1); // set zero by default

    // b vector
    double *b_vect_part = init_double_vector(lines_num[comm_rank]);
    // init part of b vector
    for (i = 0; i < lines_num[comm_rank]; ++i) {
        b_vect_part[i] = N + 1;
    }
    
    // calculating b_norm
    double b_norm;
    double b_norm_part = scalar_vector_x_vector(b_vect_part, b_vect_part, lines_num[comm_rank]);
    // collect all norm parts from processes to one variable
    MPI_Allreduce(&b_norm_part, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    b_norm = sqrt(b_norm);


    // Ax - b norm for all processes and current
    double Ax_b_norm;
    double Ax_b_norm_part;

    // buffer for new Ax-nth - b vector for this process
    double *buffer_part = init_double_vector(lines_num[comm_rank]);
    
    while (1)
    {
        int j;
        for (j = 0; j < lines_num[comm_rank]; ++j) {
            buffer_part[j] = 0;
        }

        // cycle is running until beginning rank comes back
        int curr_rank = comm_rank;
        do {
            for (j = 0; j < lines_num[curr_rank]; ++j) {
                // j * N is offset to current string in matrix_part
                // offsets[curr_rank] is offset to cell in line we need to add part we have
                int shift = j * N + offsets[curr_rank];
                buffer_part[j] += scalar_vector_x_vector(matrix_part + shift, x_vect_part, lines_num[curr_rank]);
            }

            // shift x_vect_part
            MPI_Sendrecv_replace(x_vect_part, N / comm_size + 1, MPI_DOUBLE,
                (comm_rank + 1) % comm_size, 123, (comm_rank + comm_size - 1) % comm_size, 123,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            curr_rank = (curr_rank + 1) % comm_size;
        } while (curr_rank != comm_rank);

        vector_sub_vector(buffer_part, b_vect_part, lines_num[comm_rank]);

        Ax_b_norm_part = 0;
        for(i = 0; i < lines_num[comm_rank]; ++i) {
            Ax_b_norm_part += buffer_part[i] * buffer_part[i];
        }
        
        // collect all norm parts from processes to one variable
        MPI_Allreduce(&Ax_b_norm_part, &Ax_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // don't forget to take a square root
        Ax_b_norm = sqrt(Ax_b_norm);

        if (Ax_b_norm / b_norm < EPSILON) {
            break;
        }

        vector_x_scalar(buffer_part, TAU, lines_num[comm_rank]);
        vector_sub_vector(x_vect_part, buffer_part, lines_num[comm_rank]); // now x_vect_part contains new x-n+1-th part value for this process
    }
    
    // buffer for entire new Ax-nth - b vector
    double *x_vect = init_double_vector(N);

    // collect all buffer_part from all processes to buffer
    MPI_Allgatherv(buffer_part, lines_num[comm_rank], 
        MPI_DOUBLE, x_vect, lines_num, offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    if (comm_rank == 0) {
        print_double_vector(x_vect, N);
    }

    free(lines_num);
    free(offsets);
    free(matrix_part);
    free(x_vect_part);
    free(b_vect_part);
    free(buffer_part);
    free(x_vect);
}

