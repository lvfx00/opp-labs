#pragma once

/// Solving with basic algorithm, stores 
/// x and b vectors in every MPI-process.
void calc_easy(int comm_size, int comm_rank, int N);

/// Solving with partial algorithm, separates
/// x and b vectors between MPI-processes.
void calc_hard(int comm_size, int comm_rank, int N);

