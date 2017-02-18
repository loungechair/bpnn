#pragma once

#include <vector>


namespace nn
{


typedef std::vector<double> dblvec;


namespace matrix
{

// calculate x = Ay + b
void mv_mult(double* x, const double* A, const double* y, const double* b, int rows, int cols);


// y = alpha*Ax + beta*y
// A is rows x cols
// x is cols x 1
// y is rows x 1
void mult_AXpy(double*y,
               const double alpha,
               const double* A,
               const double* x,
               const double beta,
               int rows,
               int cols);

// y += alpha*Ax
void accum_Ax(double* y, double alpha, const double* A, const double* x, int rows, int cols);

// y = alpha*ATx + beta*y
void mult_ATxpy();

// y += alpha*ATx
// A is rows x cols, so AT is cols x rows
// x is rows x 1
// y is cols x 1
void accum_ATx(double *y, double alpha, const double* A, const double* x, int rows, int cols);

// A += alpha*x*yT

// x = apha*a + beta*b

// x += alpha*y


// calculate x = a + b
void vv_add();


} // namespace matrix
} // namespace nn
