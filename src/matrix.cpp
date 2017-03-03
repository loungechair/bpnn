#include "matrix.hpp"

#include <iostream>

#ifdef _WIN32
#  include <mkl_cblas.h>
#elif __linux__
extern "C" {
#  include <cblas.h>
}
#else
#  error "Unsupported OS!"
#endif

namespace nn
{

// A += B C
template <>
void
accum_A_BC(Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C)
{
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.Rows(), A.Cols(), B.Cols(),
    1.0f, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0f, A.GetPtr(), A.Cols());
}

template <>
void
accum_A_BC(Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.Rows(), A.Cols(), B.Cols(),
    1.0, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0, A.GetPtr(), A.Cols());
}



// A += B C^T
template <>
void
accum_A_BCt(Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C)
{
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A.Rows(), A.Cols(), B.Cols(),
    1.0f, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0f, A.GetPtr(), A.Cols());
}

template <>
void
accum_A_BCt(Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A.Rows(), A.Cols(), B.Cols(),
    1.0, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0, A.GetPtr(), A.Cols());
}



// A += B^T C
template <>
void
accum_A_BtC(Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C)
{
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.Rows(), A.Cols(), B.Rows(),
    1.0f, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0f, A.GetPtr(), A.Cols());
}

template <>
void
accum_A_BtC(Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C)
{
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.Rows(), A.Cols(), B.Rows(),
    1.0, B.GetPtr(), B.Cols(), C.GetPtr(), C.Cols(), 1.0, A.GetPtr(), A.Cols());
}



// y += A^T x
template <>
void
accum_y_Atx(typename Matrix<float>::VectorType& y,
            const Matrix<float>& A,
            const typename Matrix<float>::VectorType& x)
{
  cblas_sgemv(CblasRowMajor, CblasTrans, A.Rows(), A.Cols(), 1.0f,
    A.GetPtr(), A.Cols(), &x[0], 1, 1.0f, &y[0], 1);
}

template <>
void
accum_y_Atx(typename Matrix<double>::VectorType& y,
  const Matrix<double>& A,
  const typename Matrix<double>::VectorType& x)
{
  cblas_dgemv(CblasRowMajor, CblasTrans, A.Rows(), A.Cols(), 1.0,
    A.GetPtr(), A.Cols(), &x[0], 1, 1.0, &y[0], 1);
}



// A += alpha B
template <>
void
accum_A_alphaB(Matrix<float>& A, float alpha, const Matrix<float>& B)
{
  cblas_saxpy(A.Size(), alpha, B.GetPtr(), 1, A.GetPtr(), 1);
}

template <>
void
accum_A_alphaB(Matrix<double>& A, double alpha, const Matrix<double>& B)
{
  cblas_daxpy(A.Size(), alpha, B.GetPtr(), 1, A.GetPtr(), 1);
}


}
