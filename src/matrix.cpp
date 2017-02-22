#include "matrix.hpp"

#include <iostream>

#define USE_BLAS

#ifdef USE_BLAS
#  ifdef _WIN32
#    include <mkl_cblas.h>
#  elif __linux__
extern "C" {
#    include <cblas.h>
}
#  else
#    error "Unsupported OS!"
#  endif
#endif

namespace nn
{

  
  // y += alpha*Ax
// A is rows x cols
// x is cols x 1
// y is rows x 1
void
accum_Ax(double* y, double alpha, const double* A, const double* x, int rows, int cols)
{
#ifdef USE_BLAS
  cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, A, cols, x, 1, 1.0, y, 1);
#else
  const double *Aij = A;
  double* yi = y;

  for (int r = 0; r < rows; ++r) {
    double        sum = 0;
    const double* xj  = x;

    for (int c = 0; c < cols; ++c) {
      sum += (*Aij)*(*xj);
      ++Aij;
      ++xj;
    }
    *yi += alpha * sum;
    ++yi;
  }
#endif
}

//y += alpha * Ax
void
accum_Ax(dblvector& y, double alpha, const dblvector& A, const dblvector& x, int rows, int cols)
{
#ifdef USE_BLAS
  cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, &A[0], cols, &x[0], 1, 1.0, &y[0], 1);
#endif
}

// y += alpha*ATx
// A is rows x cols, so AT is cols x rows
// x is rows x 1
// y is cols x 1
void
accum_ATx(double *y, double alpha, const double* A, const double* x, int rows, int cols)
{
#ifdef USE_BLAS
  cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, alpha, A, cols, x, 1, 1.0, y, 1);
#else
  const double* Aij = A;
  const double* xi = x;
  
  for (int r = 0; r < rows; ++r) {
    double *yj = y;
    for (int c = 0; c < cols; ++c) {
      *yj += alpha*(*Aij)*(*xi);
      ++Aij;
      ++yj;
    }
    ++xi;
  }
#endif
}

// A += alpha*xy^T
void
accum_outer_product(double* A, double alpha, const double* x, const double* y, int rows, int cols)
{
#ifdef USE_BLAS
  cblas_dger(CblasRowMajor, rows, cols, alpha, x, 1, y, 1, A, cols);
#else
  double *Aij = A;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      *(Aij++) += x[row] * y[col];
    }
  }
#endif
}


}
