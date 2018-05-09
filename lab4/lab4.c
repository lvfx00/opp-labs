#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"

// rectangle starting point
#define x0 -1.0
#define y0 -1.0
#define z0 -1.0

// rectangle length
#define Dx 2.0
#define Dy 2.0
#define Dz 2.0

// number of nodes 
#define Nx 20
#define Ny 20
#define Nz 20

//
#define APPROX 0

// wanted function
double fi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

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
    
    // fill main area
    // +1 in begin and -1 in end to skip neighbouring buffer bounds
    for(int i = 1; i < x_dim - 1; ++i) { 
        for(int j = 1; j < y_dim - 1; ++j) {
            for(int k = 0; k < z_dim - 1; ++k) {
                double coord_x = (comm_rank_x * Nx_proc + i) * hx;
                double coord_y = (comm_rank_y * Ny_proc + j) * hy;
                double coord_z = k * hz;

                // calc top border values
                if((comm_rank_y == 0 && j == 1) || 
                   (comm_rank_y == comm_size_y - 1 && j == y_dim - 2) ||
                   (comm_rank_x == 0 && i == 1) ||
                   (comm_rank_x == comm_size_x - 1 && i == x_dim - 2)) 
                {
                    printf("a");
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
    
    if(comm_rank_x == 1 && comm_rank_y == 0) {
        for(int i = 0; i < x_dim; ++i) { 
            for(int j = 0; j < y_dim; ++j) {
                printf("%lf   ", proc_nodes[i*y_dim*z_dim + j*z_dim + 7]);
            }
            printf("\n");
        }
    }

    MPI_Comm_free(&comm2d);
    MPI_Finalize();

    return 0;
}
