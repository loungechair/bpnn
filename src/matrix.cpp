#include "matrix.hpp"

#include <iostream>

namespace nn
{
namespace matrix
{

void mv_mult(double* x, const double* A, const double* y, const double* b, int rows, int cols)
{
}


// y = alpha*Ax + beta*y
// A is rows x cols
// x is cols x 1
// y is rows x 1
void mult_AXpy(double*       y,
               const double  alpha,
               const double* A,
               const double* x,
               const double  beta,
               int           rows,
               int           cols)
{
  const double* Aij = A;
  double*       yi  = y;

  
  for (int r = 0; r < rows; ++r) {
    double        sum = 0;
    const double* xj  = x;

    for (int c = 0; c < cols; ++c) {
      sum += (*Aij)*(*xj);
      ++Aij;
      ++xj;
    }
    *yi = beta * *yi + alpha * sum;
    ++yi;
  }
}


void
accum_Ax(double* y, double alpha, const double* A, const double* x, int rows, int cols)
{
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
}


// y += alpha*ATx
// A is rows x cols, so AT is cols x rows
// x is rows x 1
// y is cols x 1
void
accum_ATx(double *y, double alpha, const double* A, const double* x, int rows, int cols)
{
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
}


}
}
