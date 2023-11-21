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
    float **A;
    int **B;
    long long int N = atoll(argv[1]);
    int argumentX = atoi(argv[2]);
    A = malloc(N * sizeof(float *));
    for (long long int i = 0; i < N; i++) A[i] = malloc(N * sizeof(float));
    B = calloc(N, sizeof(int *));
    for (long long int i = 0; i < N; i++) B[i] = calloc(N, sizeof(int));

    compute1(A, B, N);
    struct L res[argumentX];
    for (int i = 0; i < argumentX; i++) {
        struct L r = compute2(B, N);
        res[i].a = r.a;
        res[i].b = r.b;
    }
    // .....
    for (long long int i = 0; i < N; ++i) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
    //.....
}