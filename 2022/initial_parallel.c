#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct L {
    long long int a;
    long long int b;
};

struct L match(int k, long long int i, long long int j) {
    struct L res = {0, 0};
    switch (k) {
        case 0:
            res.a = i - 1;
            res.b = j - 1;
            break;
        case 1:
            res.a = i - 1;
            res.b = j;
            break;
        case 2:
            res.a = i - 1;
            res.b = j + 1;
            break;
        case 3:
            res.a = i;
            res.b = j - 1;
            break;
        case 4:
            res.a = i;
            res.b = j;
            break;
        case 5:
            res.a = i;
            res.b = j + 1;
            break;
        case 6:
            res.a = i + 1;
            res.b = j - 1;
            break;
        case 7:
            res.a = i + 1;
            res.b = j;
            break;
        case 8:
            res.a = i + 1;
            res.b = j + 1;
            break;
    }
    return res;
}

void compute1(float **A, int **B, long long int N) {
    float V[9];
    for (long long int i = 1; i < N - 1; i++) {
        for (long long int j = 1; j < N - 1; j++) {
            for (int z = 0; z < 9; z++) {
                struct L tmp = match(z, i, j);
                V[z] = A[tmp.a][tmp.b];
            }
            int m = 4;
            for (int k = 0; k < 9; k++) {
                if (V[k] < V[m]) m = k;
            }
            B[i][j] = m;
        }
    }
}

struct L fnc0(int **B, long long int i, long long int j) {
    if (B[i][j] == 4) {
        struct L res = {i, j};
        return res;
    } else {
        struct L tmp = match(B[i][j], i, j);
        return fnc0(B, tmp.a, tmp.b);
    }
}

struct L compute2(int **B, long long int N) {
    int c1 = (rand() % N);
    int c2 = (rand() % N);
    struct L tmp = fnc0(B, c1, c2);
    return tmp;
}

int main(int argc, char *argv[]) {
    srand(time(0));
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ... (initialization code)

    // Determine the size of each submatrix
    long long int local_N = N / size;
    long long int excess = N % size;

    // Allocate memory for local submatrices in each process
    float **local_A = malloc((local_N + 2) * sizeof(float *));  // +2 for potential ghost rows
    int **local_B = calloc(local_N + 2, sizeof(int *));         // same here for ghost rows
    for (long long int i = 0; i < local_N + 2; i++) {
        local_A[i] = malloc(N * sizeof(float));
        local_B[i] = calloc(N, sizeof(int));
    }

    // Master process initializes the entire matrix and distributes chunks to workers
    if (rank == MASTER) {
        // ... Initialize A and B

        // Distribute chunks to each worker
        for (int p = 1; p < size; p++) {
            // Calculate the starting row for this process
            long long int start_row = p * local_N;
            if (p >= excess) start_row += excess;

            // Send the relevant chunk of A to process p
            // Include sending ghost rows if necessary for border calculations
            MPI_Send(&A[start_row][0], local_N * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }

        // Copy the master's own chunk
        for (long long int i = 0; i < local_N; i++) {
            // Copy the data
            memcpy(local_A[i + 1], A[i], N * sizeof(float));
        }
    } else {
        // Worker processes receive their chunks
        MPI_Recv(&local_A[1][0], local_N * N, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform compute1 on local chunk
    compute1(local_A, local_B, local_N, N);

    // Now perform compute2
    // This may require communication if it depends on the global state of B

    // ... (rest of the code)

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// Modify compute1 to work on submatrices
void compute1(float **local_A, int **local_B, long long int local_N, long long int N) {
    // ... (same as before, but work on local_A and local_B)
}

// ... (other function modifications as needed)
