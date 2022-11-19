#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <starpu.h>
#include <complex.h>
#include <starpu_fxt.h>
//#include <common/blas.h>
//#include <cblas.h>


//#if defined(STARPU_GOTO) || defined(STARPU_SYSTEM_BLAS) || defined(STARPU_MKL)

extern double dnrm2_(int const *, double const *, int const *);

extern void dtrmm_(char const *, char const *, char const *, char const *,
    int const *, int const *, double const *, double const *, int const *,
    double *, int const *);

extern void dlacpy_(char const *, int const *, int const *, double const *,
    int const *, double *, int const *);

extern double dlange_(char const *, int const *, int const *, double const *,
    int const *, double *);

extern void dtrsm_(char const *, char const *, char const *, char const *,
    int const *, int const *, double const *, double const *, int const *,
    double *, int const *);

extern void dgemm_(char const *, char const *, int const *, int const *,
    int const *, double const *, double const *, int const *, double const *,
    int const *, double const *, double *, int const *);
/*
#endif
*/

double one = 1.0;
double minus_one = -1.0;

void init_matrix(double *A, int pb, int qb)
{
    for (int i = 0; i < pb; i++)
    {
        int j = 0;
        for (; j <= i; j++)
            A[i * qb + j] = j + 1;
        for (; j < qb; j++)
            A[i * qb + j] = i + 1;
    }

}
void Print(int row, int col, double *A)
{
    printf("       ");

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%10f ", A[i * col + j]);
        }
        printf("\n");
        printf("       ");
    }
    printf("*****************************************************************************************************\n");
}


int DIVCEIL(int a, int b)
{
    return (a+b-1)/b;
}

int MIN(int a, int b)
{
    return a < b ? a : b;
}

int MAX(int a, int b)
{
    return a > b ? a : b;
}

void simple_lu(int n, int ldA, double *A)
{
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            A[i*ldA+j] /= A[i*ldA+i];

            for (int k = i+1; k < n; k++)
                A[k*ldA+j] -= A[i*ldA+j] * A[k*ldA+i];
        }
    }
}

static void small_lu(void *buffers[], void *args)
{
    // In this case the kernel has a single input and output buffer. The buffer
    // is accessible through a matrix interface.
    struct starpu_matrix_interface *A_i =
        (struct starpu_matrix_interface *)buffers[0];

    // we can now extract the relevant information from the interface
    double *ptr = (double *) STARPU_MATRIX_GET_PTR(A_i); // pointer
    const int n = STARPU_MATRIX_GET_NX(A_i);             // matrix dimension
    const int ld = STARPU_MATRIX_GET_LD(A_i);            // leading dimension

    // The runtime system guarantees that the data resides in the device memory
    // (main memory in this case). Thus, we can call the simple_lu function to
    // perform the actual computations.
    simple_lu(n, ld, ptr);
}

// a CPU implementation for the kernel that performs a block row/column update
static void rc_update(void *buffers[], void *args)
{
    // The first four dtrsm arguments are passed as statix arguments. This
    // allows us to use the same codelet to perform the block row and block
    // column updates.
    char side, uplo, transa, diag;
    starpu_codelet_unpack_args(args, &side, &uplo, &transa, &diag);

    // This time we have two buffers:
    //   0 = a small LU decomposition that corresponds to the diagonal block
    //   1 = current row/column block
    //
    // Note that we do not have define the interface explicitly.

    dtrsm_(&side, &uplo, &transa, &diag,
        (int *)&STARPU_MATRIX_GET_NX(buffers[1]),
        (int *)&STARPU_MATRIX_GET_NY(buffers[1]),
        &one,
        (double *)STARPU_MATRIX_GET_PTR(buffers[0]),
        (int *)&STARPU_MATRIX_GET_LD(buffers[0]),
        (double *)STARPU_MATRIX_GET_PTR(buffers[1]),
        (int *)&STARPU_MATRIX_GET_LD(buffers[1]));

}

// a CPU implementation for the kernel that performs a trailing matrix update
static void trail_update(void *buffers[], void *args)
{
    // This time we have three buffers:
    //  0 = corresponding column block
    //  1 = corresponding row block
    //  2 = current trailing matrix block

     dgemm_("No Transpose", "No Transpose",
        (int *)&STARPU_MATRIX_GET_NX(buffers[2]),
        (int *)&STARPU_MATRIX_GET_NY(buffers[2]),
        (int *)&STARPU_MATRIX_GET_NY(buffers[0]),
        &minus_one,
        (double *)STARPU_MATRIX_GET_PTR(buffers[0]),
        (int *)&STARPU_MATRIX_GET_LD(buffers[0]),
        (double *)STARPU_MATRIX_GET_PTR(buffers[1]),
        (int *)&STARPU_MATRIX_GET_LD(buffers[1]),
        &one,
        (double *)STARPU_MATRIX_GET_PTR(buffers[2]),
        (int *)&STARPU_MATRIX_GET_LD(buffers[2]));
}

//
// Codelets
//
//  A codelet encapsulates the various implementations of a computational
//  kernel.
//

// a codelet that computes a small LU decomposition
static struct starpu_codelet small_lu_cl = {
    .name = "small_lu",                 // codelet name
    .cpu_funcs = { small_lu },          // pointers to the CPU implementations
    .nbuffers = 1,                      // buffer count
    .modes = { STARPU_RW }              // buffer access modes (read-write)
};

// a codelet that that performs a block row/column update
static struct starpu_codelet rc_update_cl = {
    .name = "rc_update",
    .cpu_funcs = { rc_update },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }    // read-only, read-write
};

// a codelet that performs a trailing matrix update
static struct starpu_codelet trail_update_cl = {
    .name = "trail_update",
    .cpu_funcs = { trail_update },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};


void blocked_lu(int block_size, int n, int ldA, double *A)
{
    const int block_count = DIVCEIL(n, block_size);

    // initialize StarPU
    int ret = starpu_init(NULL);

    if (ret != 0)
        return;

    // Each buffer that is to be passed to a task must be encapsulated inside a
    // data handle. This means that we must allocate and fill an array that
    // stores the block handles.

    starpu_data_handle_t **blocks =
        malloc(block_count*sizeof(starpu_data_handle_t *));

 for (int i = 0; i < block_count; i++) {
        blocks[i] = malloc(block_count*sizeof(starpu_data_handle_t));

        for (int j = 0; j < block_count; j++) {
            // each block is registered as a matrix
            starpu_matrix_data_register(
                &blocks[i][j],                      // handle
                STARPU_MAIN_RAM,                    // memory node
                (uintptr_t)(A+(j*ldA+i)*block_size), // pointer
                ldA,                                 // leading dimension
                MIN(block_size, n-i*block_size),    // row count
                MIN(block_size, n-j*block_size),    // column count
                sizeof(double));                    // element size
        }
    }
    struct timeval start, end; 

    gettimeofday(&start, NULL);
    double time = 0;
    // go through the diagonal blocks
    for (int i = 0; i < block_count; i++) {

        // insert a task that processes the current diagonal block
        starpu_task_insert(
            &small_lu_cl,       // codelet
            STARPU_PRIORITY,    // the next argument specifies the priority
            STARPU_MAX_PRIO,    // priority
            STARPU_RW,          // the next argument is a read-write handle
            blocks[i][i],       // handle to the diagonal block
            0);                 // a null pointer finalizes the call

        // insert tasks that process the blocks to the right of the current
        // diagonal block
        for (int j = i+1; j < block_count; j++) {

            // blocks[i][j] <- L1(blocks[i][i]) \ blocks[i][j]
            starpu_task_insert(&rc_update_cl,
                STARPU_PRIORITY, MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-j+i),
                STARPU_VALUE,   // the next argument is a static argument
                "Left",         // pointer to the static argument
                sizeof(char),   // size of the static argument
                STARPU_VALUE, "Lower", sizeof(char),
                STARPU_VALUE, "No transpose", sizeof(char),
                STARPU_VALUE, "Unit triangular", sizeof(char),
                STARPU_R, blocks[i][i],
                STARPU_RW, blocks[i][j], 0);
        }

        // insert tasks that process the blocks below the current diagonal block
        for (int j = i+1; j < block_count; j++) {

            // blocks[j][i] <- U(blocks[i][i]) / blocks[j][i]
            starpu_task_insert(&rc_update_cl,
                STARPU_PRIORITY, MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-j+i),
                STARPU_VALUE, "Right", sizeof(char),
                STARPU_VALUE, "Upper", sizeof(char),
                STARPU_VALUE, "No transpose", sizeof(char),
                STARPU_VALUE, "Not unit triangular", sizeof(char),
                STARPU_R, blocks[i][i],
                STARPU_RW, blocks[j][i], 0);
        }

        // insert tasks that process the trailing matrix
        for (int ii = i+1; ii < block_count; ii++) {
            for (int jj = i+1; jj < block_count; jj++) {

                // blocks[ii][jj] <-
                //               blocks[ii][jj] - blocks[ii][i] * blocks[i][jj]
                starpu_task_insert(&trail_update_cl,
                    STARPU_PRIORITY, MAX(
                        MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-ii+i),
                        MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-jj+i)),
                    STARPU_R, blocks[ii][i],
                    STARPU_R, blocks[i][jj],
                    STARPU_RW, blocks[ii][jj], 0);
            }
        }
    }
   gettimeofday(&end, NULL);
    time += (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec-start.tv_usec)/1000.0;     
    printf("The total time is %.2lf ms\n", time);
    for (int i = 0; i < block_count; i++) {
        for (int j = 0; j < block_count; j++) {
            starpu_data_unregister(blocks[i][j]);
        }
        free(blocks[i]);
    }
    free(blocks);

    starpu_shutdown();
}



int main(int argc, char **argv)
{
    int N = 100;
    int block_size = 10;

     double *A = (double *)malloc(sizeof(double) * N * N); 
     double *B = (double *)malloc(sizeof(double) * N * N); 

    init_matrix(A,N,N);
    init_matrix(B,N,N);
    //Print(N,N,A);


    blocked_lu(block_size, N, N, A);
 
    //Print(N,N,A);
 
}
