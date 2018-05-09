#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"

#define APPROX 0

// wanted function
double fi(double x, double y, double z) {
    return x*x + y*y + z*z;
}

int main(int argc, char **argv) {
    // input data
    
    // rectangle starting point
    static const double x0 = -1.0;
    static const double y0 = -1.0;
    static const double z0 = -1.0;

    // rectangle sides
    static const double Dx = 2.0;
    static const double Dy = 2.0;
    static const double Dz = 2.0;

    // number of nodes 
    static const int Nx = 20;
    static const int Ny = 20;
    static const int Nz = 20;

    ////////////////////////////////////////////////////////////////////////////////
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
    int comm_rank_y = coords[0]; // from 0 to comm_size_y - 1
    int comm_rank_x = coords[1];

    ////////////////////////////////////////////////////////////////////////////////
    
    // number of nodes per one process
    int Nx_proc = Nx / comm_size_x;
    int Ny_proc = Ny / comm_size_y;
    int Nz_proc = Nz;

    // distances between nodes
    double hx = Dx / (Nx - 1);
    double hy = Dy / (Ny - 1);
    double hz = Dz / (Nz - 1);

    // create array of nodes for each process
    int x_dim = Nx_proc + 2; // +2 for neighbouring processes buffer bounds
    int y_dim = Ny_proc + 2;
    int z_dim = Nz;
    double proc_nodes[x_dim * y_dim * z_dim];

    // fill array by initial values:
    // F(x, y, z) at borders - known values of wanted function fi(z, y, z)
    // fi0th(x, y, z) on main area - beginning approximation
    
    // +1 in begin and -1 in end to skip neighbouring buffer bounds
    for(int i = 1; i < x_dim - 1; ++i) { 
        for(int j = 1; j < y_dim - 1; ++j) {
            for(int k = 0; k < z_dim - 1; ++k) {
                double coord_x = x0 + (comm_rank_x * Nx_proc + i - 1) * hx;
                double coord_y = y0 + (comm_rank_y * Ny_proc + j - 1) * hy;
                double coord_z = z0 + k * hz;

                if(coord_x > 1 || coord_y > 1 || coord_z > 1) {
                    printf("achtung!!!\n");
                }

                // check if we on border node:
                    // top border
                if((comm_rank_y == 0 && j == 1) || 
                    // bottom border
                   (comm_rank_y == comm_size_y - 1 && j == y_dim - 2) ||
                    // left border
                   (comm_rank_x == 0 && i == 1) ||
                    // right border
                   (comm_rank_x == comm_size_x - 1 && i == x_dim - 2)) 
                {
                    proc_nodes[i*y_dim*z_dim + j*z_dim + k] 
                        = fi(coord_x, coord_y, coord_z);
                } else {
                    proc_nodes[i*y_dim*z_dim + j*z_dim + k] = APPROX;
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // swap neighbouring buffer bounds
    
    // create type for left and right borders
    MPI_Datatype row_type;
    MPI_Type_contiguous((y_dim-2)*z_dim, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // create type for top and bottom borders
    MPI_Datatype temp_type, col_type;
    MPI_Type_vector(x_dim-2, z_dim, y_dim*z_dim, MPI_DOUBLE, &temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(double) * z_dim, &col_type);
    MPI_Type_commit(&col_type);

    int next_x, prev_x, next_y, prev_y;
    MPI_Cart_shift(comm2d, 0, 1, &prev_y, &next_y);
    MPI_Cart_shift(comm2d, 1, 1, &prev_x, &next_x);

    // send top border to upper proc
    if(comm_rank_y != 0) {
        MPI_Send(proc_nodes + z_dim*y_dim + z_dim,
                1, col_type, prev_y, 0, comm2d);
    }
    // send bottom border to lower proc
    if(comm_rank_y != comm_size_y - 1) {
        MPI_Send(proc_nodes + 2*z_dim*y_dim - 2*z_dim,
                1, col_type, next_y, 0, comm2d);
    }
    // send left border to <-
    if(comm_rank_x != 0) {
        MPI_Send(proc_nodes + z_dim*y_dim + z_dim,
                1, row_type, prev_x, 0, comm2d);
    }
    // send right border to ->
    if(comm_rank_x != comm_size_x - 1) {
        MPI_Send(proc_nodes + z_dim*y_dim*(x_dim-2)+ z_dim,
                1, row_type, next_x, 0, comm2d);

    }

    // recieve top neighbour bound from top
    if(comm_rank_y != 0) {
        MPI_Recv(proc_nodes + z_dim*y_dim,
                1, col_type, prev_y, 0, comm2d, MPI_STATUS_IGNORE);
    }
    // recieve bottom neighbour bound from down
    if(comm_rank_y != comm_size_y - 1) {
        MPI_Recv(proc_nodes + z_dim*y_dim*2 - z_dim,
                1, col_type, next_y, 0, comm2d, MPI_STATUS_IGNORE);
    }
    // recieve left neighbour bound <-
    if(comm_rank_x != 0) {
        MPI_Recv(proc_nodes + z_dim,
                1, row_type, prev_x, 0, comm2d, MPI_STATUS_IGNORE);
    }
    // recieve right neighbour bound from ->
    if(comm_rank_x != comm_size_x - 1) {
        MPI_Recv(proc_nodes + z_dim*y_dim*(x_dim-1) + z_dim,
                1, row_type, next_x, 0, comm2d, MPI_STATUS_IGNORE);
    }


    ////////////////////////////////////////////////////////////////////////////////
    
    if(comm_rank_x == 1 && comm_rank_y == 1) {
        for(int i = 0; i < y_dim; ++i) { 
            for(int j = 0; j < x_dim; ++j) {
                printf("%lf   ", proc_nodes[j*y_dim*z_dim + i*z_dim + 0]);
            }
            printf("\n");
        }
    }

    MPI_Comm_free(&comm2d);
    MPI_Finalize();

    return 0;
}
