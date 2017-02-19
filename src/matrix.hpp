#pragma once

#include <vector>


namespace nn
{

typedef std::vector<double> dblvec;

namespace matrix
{

// y += alpha*Ax
void accum_Ax(double* y, double alpha, const double* A, const double* x, int rows, int cols);

// y += alpha*ATx
// A is rows x cols, so AT is cols x rows
// x is rows x 1
// y is cols x 1
void accum_ATx(double *y, double alpha, const double* A, const double* x, int rows, int cols);

void
accum_outer_product(double* A, double alpha, const double* x, const double* y, int rows, int cols);


} // namespace matrix
} // namespace nn
