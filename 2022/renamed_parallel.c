#include <mpi.h>
#include <stdlib.h>
#include <time.h>

struct Coordinates {
    long long int x;
    long long int y;
};

struct Coordinates getNeighbor(int direction, long long int x, long long int y) {
    struct Coordinates coord = {0, 0};
    switch (direction) {
        case 0:
            coord.x = x - 1;
            coord.y = y - 1;
            break;
        case 1:
            coord.x = x - 1;
            coord.y = y;
            break;
        case 2:
            coord.x = x - 1;
            coord.y = y + 1;
            break;
        case 3:
            coord.x = x;
            coord.y = y - 1;
            break;
        case 4:
            coord.x = x;
            coord.y = y;
            break;
        case 5:
            coord.x = x;
            coord.y = y + 1;
            break;
        case 6:
            coord.x = x + 1;
            coord.y = y - 1;
            break;
        case 7:
            coord.x = x + 1;
            coord.y = y;
            break;
        case 8:
            coord.x = x + 1;
            coord.y = y + 1;
            break;
    }
    return coord;
}

void findLocalMinima(float **gridValues, int **minimaPath, long long int size) {
    float neighbors[9];
#pragma omp parallel for private(neighbors)
    for (long long int x = 1; x < size - 1; x++) {
        for (long long int y = 1; y < size - 1; y++) {
            for (int z = 0; z < 9; z++) {
                struct Coordinates neighborCoord = getNeighbor(z, x, y);
                neighbors[z] = gridValues[neighborCoord.x][neighborCoord.y];
            }
            int minIndex = 4;  // Start with the current cell
            for (int k = 0; k < 9; k++) {
                if (neighbors[k] < neighbors[minIndex]) minIndex = k;
            }
            minimaPath[x][y] = minIndex;
        }
    }
}

struct Coordinates tracePathToMinima(int **minimaPath, long long int x, long long int y) {
    if (minimaPath[x][y] == 4) {  // Current cell is the local minimum
        struct Coordinates result = {x, y};
        return result;
    } else {
        struct Coordinates nextCoord = getNeighbor(minimaPath[x][y], x, y);
        return tracePathToMinima(minimaPath, nextCoord.x, nextCoord.y);
    }
}

struct Coordinates findRandomMinima(int **minimaPath, long long int size) {
    int randomX = rand() % size;
    int randomY = rand() % size;
    return tracePathToMinima(minimaPath, randomX, randomY);
}

__global__ void findLocalMinimaKernel(float *d_gridValues, int *d_minimaPath, long long int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < size - 1 && y > 0 && y < size - 1) {
        float neighbors[9];
        // Populate neighbors array and find local minima as in the original function
        // Remember to use linear indexing for d_gridValues and d_minimaPath
        // ...
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int size = atoll(argv[1]);  // Assuming size is divisible by world_size
    long long int local_size = size / world_size;
    int numPaths = atoi(argv[2]);

    float **d_gridValues;
    int **d_minimaPath;

    // Allocate memory on the device for d_gridValues and d_minimaPath
    cudaMalloc(&d_gridValues, size * size * sizeof(float));
    cudaMalloc(&d_minimaPath, size * size * sizeof(int));

    cudaMemcpy(d_gridValues, gridValues, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_minimaPath, 0, size * size * sizeof(int));  // Initialize minimaPath on device

    float **local_gridValues;
    int **local_minimaPath;

    // Allocate memory for each process's portion of the grid
    // Each process handles a slice of the grid
    local_gridValues = malloc(local_size * sizeof(float *));
    local_minimaPath = calloc(local_size, sizeof(int *));
    for (long long int i = 0; i < local_size; i++) {
        local_gridValues[i] = malloc(size * sizeof(float));
        local_minimaPath[i] = calloc(size, sizeof(int));
    }

    // Master process initializes the full grid and distributes it to the workers
    if (world_rank == 0) {
        // Master initializes the full grid
        gridValues = malloc(size * sizeof(float *));
        for (long long int i = 0; i < size; i++) {
            gridValues[i] = malloc(size * sizeof(float));
            // Initialize gridValues[i] here
        }

        // Copy the first slice to the master's local grid
        for (long long int i = 0; i < local_size; i++) {
            memcpy(local_gridValues[i], gridValues[i], size * sizeof(float));
        }

        // Distribute slices of the grid to each worker
        for (int i = 1; i < world_size; i++) {
            for (long long int j = 0; j < local_size; j++) {
                MPI_Send(gridValues[i * local_size + j], size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

        // Master can now free the full grid as it's no longer needed
        for (long long int i = 0; i < size; i++) {
            free(gridValues[i]);
        }
        free(gridValues);
    } else {
        // Workers receive their slice of the grid
        // Use MPI_Recv to receive the slice
        for (long long int i = 0; i < local_size; i++) {
            MPI_Recv(local_gridValues[i], size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Both master and workers perform local minima finding on their slice
    findLocalMinima(local_gridValues, local_minimaPath, local_size);

    if (world_rank != 0) {
        // Workers send their results back to master
        // Assuming that we are sending back an array of some computed results
        // For demonstration, let's say we send back the local_minimaPath
        for (long long int i = 0; i < local_size; i++) {
            MPI_Send(local_minimaPath[i], size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        // Master collects results from all workers
        int **collectedResults = calloc(size, sizeof(int *));
        for (long long int i = 0; i < local_size; i++) {
            collectedResults[i] = local_minimaPath[i];  // The master's own results
        }

        for (int i = 1; i < world_size; i++) {
            for (long long int j = 0; j < local_size; j++) {
                collectedResults[i * local_size + j] = calloc(size, sizeof(int));
                MPI_Recv(collectedResults[i * local_size + j], size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Master aggregates and processes all results
        // ... [Processing of results by the master] ...

        // Freeing the collected results
        for (long long int i = local_size; i < size; i++) {
            free(collectedResults[i]);
        }
        free(collectedResults);
    }

    // Free the allocated memory
    for (long long int i = 0; i < local_size; ++i) {
        free(local_gridValues[i]);
        free(local_minimaPath[i]);
    }
    free(local_gridValues);
    free(local_minimaPath);

    MPI_Finalize();
    return 0;
}
