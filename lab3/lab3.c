#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"

#define n1 32
#define n2 4
#define n3 16

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int dims[2] = {0, 0};
    MPI_Dims_create(comm_size, 2, dims);
    int comm_size_y = dims[0];
    int comm_size_x = dims[1];

    MPI_Comm comm2d;
    int periods[2] = {0, 0};
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);

    // get process number in comm2d communicator
    int comm_rank;
    MPI_Comm_rank(comm2d, &comm_rank);

    // get process coordinates in 2d proc matrix
    int coords[2];
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    int comm_rank_y = coords[0];
    int comm_rank_x = coords[1];

    // create matrxes in (0, 0) process
    double *A;
    double *B;

    if(comm_rank_x == 0 && comm_size_y == 0) {
        A = malloc(n1 * n2 * sizeof(double));
        assert(A != NULL);
        // fill the matrix
        for(int i = 0; i < n1; ++i) {
            for(int j = 0; j < n2; ++j) {
                A[i * n2 + j] = 1000 * i + j;
            }
        }

        B = malloc(n2 * n3 * sizeof(double));
        assert(B != NULL);
        // fill the matrix
        for(int i = 0; i < n2; ++i) {
            for(int j = 0; j < n3; ++j) {
                B[i * n3 + j] = 1000 * i + j;
            }
        }
    }

    // split comm2d into horizontal lines communicators
    int color = comm_rank_y;
    int key = comm_rank_x;

    MPI_Comm row_comm;
    MPI_Comm_split(comm2d, color, key, &row_comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    int row_size;
    MPI_Comm_size(row_comm, &row_size); 
    
    // split comm2d into vertical lines communicators
    color = comm_rank_x;
    key = comm_rank_y;

    MPI_Comm col_comm;
    MPI_Comm_split(comm2d, color, key, &row_comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);
    int col_size;
    MPI_Comm_size(col_comm, &col_size);

    printf("CARD X/Y: %d/%d \t ROW RANK/SIZE: %d/%d\n", comm_rank_x, comm_rank_y, row_rank, row_size);
    
    /*
    // create type for A matrix row
    MPI_Datatype row_type;
    MPI_Type_contiguous(n2, MPI_DOUBLE, &row_type); // n2 - num of elements in row
    MPI_Type_commit(&row_type);


    ///////////////////////////////////////////////////////////////////////
    // number of rows per one processor
    int row_num = n1 / comm_size_y; 

    // space for row_num lines with n2 elements
    double *A_part = (double*)malloc(row_num * n2 * sizeof(double));
    assert(A_part != NULL);
    
    // scatter A matrix rows between x=0 processes (first column)
        
    // number of sending elements for each process
    int * sendcounts = (int *)malloc(comm_size_y * sizeof(int));
    assert(sendcounts != NULL);
    for(int i = 0; i < comm_size_y; ++i) {
        sendcounts[i] = row_num;
    }

    // displacements of sending data for each process
    int * displs = (int *)malloc(comm_size_y * sizeof(int));
    assert(displs != NULL);
    displs[0] = 0;
    for(int i = 1; i < comm_size_y; ++i) {
        displs[i] = displs[i - 1] + row_num;
    }

    if(comm_rank_x == 0) {
        MPI_Scatterv(A, sendcounts, displs, row_type, A_part, row_num, row_type, 0, col_comm); 
    }

    if(comm_rank_x == 0 && comm_rank_y == 0) {
        for(int i = 0; i < row_num * n2; ++i)
        printf("%lf   ", A_part[i]);
    }

    MPI_Datatype col_type;
    MPI_Type_vector(n2, 1, n1, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    */

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&comm2d);
    MPI_Finalize();
    return 0;
}
