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

int main(int argc, char *argv[]) {
    srand(time(0));
    float **gridValues;
    int **minimaPath;
    long long int size = atoll(argv[1]);
    int numPaths = atoi(argv[2]);

    // Memory allocation for gridValues and minimaPath
    gridValues = malloc(size * sizeof(float *));
    minimaPath = calloc(size, sizeof(int *));
    for (long long int i = 0; i < size; i++) {
        gridValues[i] = malloc(size * sizeof(float));
        minimaPath[i] = calloc(size, sizeof(int));
    }

    findLocalMinima(gridValues, minimaPath, size);

    struct Coordinates minimaCoordinates[numPaths];
    for (int i = 0; i < numPaths; i++) {
        minimaCoordinates[i] = findRandomMinima(minimaPath, size);
    }

    // Cleanup code for gridValues and minimaPath
    for (long long int i = 0; i < size; ++i) {
        free(gridValues[i]);
        free(minimaPath[i]);
    }
    free(gridValues);
    free(minimaPath);

    // Rest of the main function...
}
