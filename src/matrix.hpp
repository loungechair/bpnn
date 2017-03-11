#pragma once

#include <vector>
#include <array>

#include <mkl_cblas.h>


namespace nn
{
template <typename T> class Matrix;

typedef double dblscalar;
typedef std::vector<dblscalar> dblvector;
typedef Matrix<dblscalar> dblmatrix;

template <typename T>
class Matrix
{
public:
  
  typedef T ValueType;
  typedef std::vector<T> VectorType;
  typedef typename VectorType::iterator IteratorType;
  typedef typename VectorType::const_iterator ConstIteratorType;


  class RowType
  {
  public:
    RowType(std::pair<IteratorType, IteratorType> range)
      : begin_iterator(range.first),
        end_iterator(range.second)
    {}
    IteratorType begin() { return begin_iterator; }
    IteratorType end() { return end_iterator; }
  private:
    IteratorType begin_iterator;
    IteratorType end_iterator;
  };


  Matrix(int rows_use, int cols_use)
    : rows(rows_use),
      cols(cols_use),
      size(rows*cols),
      data(size, 0)
  {
  }


  Matrix(const std::vector<VectorType>& v)
  {
    rows = v.size();
    cols = v[0].size();
    for (auto& pattern : v) {
      if (pattern.size() != cols) {
        std::cerr << "Pattern is wrong size!" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    size = rows * cols;
    data.resize(size);
    for (int row = 0; row < rows; ++row) {
      SetRowValues(row, v[row]);
    }
  }

  int GetRowStartIndex(int row_num) const
  {
    return row_num * cols;
  }

  std::pair<IteratorType, IteratorType> GetRowRange(int row_num)
  {
    auto start_iterator = data.begin() + row_num * cols;
    auto end_iterator = start_iterator + cols;
    return std::make_pair(start_iterator, end_iterator);
  }

  RowType GetRow(int row_num) { return RowType(GetRowRange(row_num)); }

  const RowType GetRow(int row_num) const { return RowType(GetRowRange(row_num)); }

  VectorType GetRowValues(int row_num)
  {
    auto range = GetRowRange(row_num);
    return dblvector(range.first, range.second);
  }

  VectorType GetColumnValues(int col_num)
  {
    dblvector values(cols);
    auto v = data.begin() + col_num;
    for (int c = 0; c < cols; ++c, v += cols) {
      values[c] = *v;
    }
    return values;
  }

  void SetRowValues(int row_num, const dblvector& values)
  {
    std::copy(std::begin(values), std::end(values), data.begin() + row_num * cols);
  }

  void SetData(const dblvector& values)
  {
    if (values.size() != size) {
      std::cerr << "Invalid size" << std::endl;
    }

    std::copy(std::begin(values), std::end(values), std::begin(data));
  }

  void SetEntry(int row, int col, T value) { data[row * cols + col] = value; }
  void SetEntry(int index, T value) { data[index] = value; }

  T& operator[](int index) { return data[index]; }
  const T& operator[](int index) const { return data[index]; }

  void SetAllRowValues(const dblvector& values)
  {
    for (int i = 0; i < rows; ++i) {
      SetRowValues(i, values);
    }
  }

  int Rows() const { return rows; }
  int Cols() const { return cols; }
  int Size() const { return size; }

  T* GetPtr() { return &data[0]; }
  const T* GetPtr() const { return &data[0]; }
  std::vector<T>& GetRef() { return data; }
  const std::vector<T>& GetRef() const { return data; }

  IteratorType begin() { return data.begin(); }
  IteratorType end()   { return data.end(); }
  ConstIteratorType begin() const { return data.begin(); }
  ConstIteratorType end()   const { return data.end(); }

  T Norm() const {
    return cblas_dnrm2(data.size(), &data[0], 1);
  }

  void Normalize() {
    T norm = Norm();
    if (norm > 1.0) {
      cblas_dscal(data.size(), 1.0 / norm, &data[0], 1);
    }
  }

  T* GetRowPtr(int row_num)
  {
    return &data[0] + GetRowStartIndex(row_num);
  }

  void NormalizeRow(int row_num, T desired_norm = 1.0)
  {
    T row_norm = cblas_dnrm2(cols, GetRowPtr(row_num), 1);
    T scale_factor = desired_norm / row_norm;
    cblas_dscal(cols, scale_factor, GetRowPtr(row_num), 1);
  }

  void NormalizeEachRow(T desired_norm = 1.0)
  {
    for (int row = 0; row < rows; ++row) {
      NormalizeRow(row, desired_norm);
    }
  }
  
  void print()
  {
    int idx = 0;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << data[idx++] << "\t";
      }
      std::cout << std::endl;
    }
  }

private:
  int rows;
  int cols;
  int size;

  std::vector<T> data;
};



template <typename T> typename Matrix<T>::IteratorType begin(Matrix<T>& A) { return A.begin(); }
template <typename T> typename Matrix<T>::ConstIteratorType begin(const Matrix<T>& A) { return A.begin(); }
template <typename T> typename Matrix<T>::IteratorType end(Matrix<T>& A) { return A.end(); }
template <typename T> typename Matrix<T>::ConstIteratorType end(const Matrix<T>& A) { return A.end(); }



// matrix-matrix operations
// A += B C
template <typename T>
void accum_A_BC(Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C);


// A += B C^T
template <typename T>
void accum_A_BCt(Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C);


// A += B^T C
template <typename T>
void accum_A_BtC(Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C);


// y += A^T x
template <typename T>
void accum_y_Atx(typename Matrix<T>::VectorType& y, const Matrix<T>& A,
  const typename Matrix<T>::VectorType& x);


// A += alpha B
template <typename T>
void accum_A_alphaB(Matrix<T>& A, T alpha, const Matrix<T>& B);


// y += alpha x
template <typename T>
void accum_y_alphax(std::vector<T>& y, T alpha, const std::vector<T>& x);


// A += x y^T
template <typename T>
void accum_A_xyT(Matrix<T>& A, const typename Matrix<T>::VectorType& x, const typename Matrix<T>::VectorType& y);

} // namespace nn
