// c header 
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
// cpp header
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// 
#include <gflags/gflags.h>
// science computation lib.
#include <mkl_cblas.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include <yael/kmeans.h>
#include <yael/vector.h>
#include <yael/matrix.h>

int fvecs_read(const char *fname, int d, int n, float *v);
int fvec_read(const char *fname, int d, float *a, int o_f);
int ivecs_new_read(const char *fname, int *d_out, int **vi);
int *ivec_new_read(const char *fname, int *d_out);
int b2fvecs_read(const char *fname, int d, int n, float *v);
void fvec_add(float *v1, const float *v2, long n);
void fvec_sub(float *v1, const float *v2, long n);
float *fmat_new_transp(const float *a, int ncol, int nrow);
void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k, const char *transp, float *result);
void fmat_rev_subtract_from_columns(int d, int n, float *m, const float *avg);

#ifdef __cplusplus
}
#endif

#define THREADS_POOL_JOIN(THREADS_COUNT, THREADS_POOL) \
  for (auto curr_thread_id = 0; curr_thread_id < THREADS_COUNT; ++curr_thread_id) { \
    THREADS_POOL[curr_thread_id].join(); \
  }

#define THREADS_POOL_ASSIGN(THREADS_COUNT, THREADS_POOL, THREADS_FUNC) \
  THREADS_POOL.clear(); \
  THREADS_POOL.reserve(THREADS_COUNT); \
  for (auto curr_thread_id = 0; curr_thread_id < THREADS_COUNT; ++curr_thread_id) { \
    THREADS_POOL.push_back(std::thread(THREADS_FUNC, curr_thread_id)); \
  }
