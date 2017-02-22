#pragma once

#include <vector>
#include <array>


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
  
  typedef std::vector<T> VectorType;
  typedef typename std::vector<T>::iterator IteratorType;
  typedef T ValueType;


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

  std::pair<IteratorType, IteratorType> GetRowRange(int row_num)
  {
    auto start_iterator = data.begin() + row_num * cols;
    auto end_iterator = start_iterator + cols;
    return std::make_pair(start_iterator, end_iterator);
  }

  RowType GetRow(int row_num) { return RowType(GetRowRange(row_num)); }

  dblvector GetRowValues(int row_num)
  {
    auto range = GetRowRange();
    return dblvector(range.first, range.second);
  }

  dblvector GetColumnValues(int col_num)
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
    std::copy(std::begin(values), std::end(values), GetRowRange(row_num).first);
  }

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
  T& GetRef() { return data; }
  const T& GetRef() const { return data; }

  IteratorType begin() { return data.begin(); }
  IteratorType end() { return data.end(); }

private:
  int rows;
  int cols;
  int size;

  std::vector<T> data;
};


// y += alpha*Ax
void accum_Ax(double* y, double alpha, const double* A, const double* x, int rows, int cols);

void accum_Ax(dblvector& y, double alpha, const dblvector& A, const dblvector& x, int rows, int cols);

// y += alpha*ATx
// A is rows x cols, so AT is cols x rows
// x is rows x 1
// y is cols x 1
void accum_ATx(double *y, double alpha, const double* A, const double* x, int rows, int cols);

void
accum_outer_product(double* A, double alpha, const double* x, const double* y, int rows, int cols);


} // namespace nn
