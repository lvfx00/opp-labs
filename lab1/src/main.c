#include <mpi.h>
#include <stdio.h>
#include "calc.h"

#define N 1100

void main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    calc_easy(comm_size, comm_rank, N);
    calc_hard(comm_size, comm_rank, N);

    MPI_Finalize();
}
