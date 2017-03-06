#include "gtest/gtest.h"

#include "../src/matrix.hpp"

nn::dblmatrix CreateMatrix(int rows, int cols)
{
  nn::dblmatrix A(rows, cols);

  for (int i = 0; i < A.Size(); ++i) {
    A.SetEntry(i, i + 1);
  }

  return A;
}

TEST(Matrix, accum_A_BC)
{
  auto B = CreateMatrix(2, 3);
  auto C = CreateMatrix(3, 2);
  nn::dblmatrix A(2, 2);

  nn::accum_A_BC(A, B, C);

  nn::dblvector answer{ 22, 28, 49, 64 };
  const auto& result = A.GetRef();
  
  EXPECT_EQ(answer.size(), result.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(answer[i], result[i]);
  }

  nn::accum_A_BC(A, B, C);

  const auto& result2 = A.GetRef();

  EXPECT_EQ(answer.size(), result2.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(2*answer[i], result[i]);
  }
}


TEST(Matrix, accum_A_BCt)
{
  auto B = CreateMatrix(2, 3);
  auto C = CreateMatrix(2, 3);
  nn::dblmatrix A(2, 2);

  nn::accum_A_BCt(A, B, C);

  nn::dblvector answer{ 14, 32, 32, 77 };
  const auto& result = A.GetRef();

  EXPECT_EQ(answer.size(), result.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(answer[i], result[i]);
  }
}


TEST(Matrix, accum_A_BtC)
{
  auto B = CreateMatrix(3, 2);
  auto C = CreateMatrix(3, 2);
  nn::dblmatrix A(2, 2);

  nn::accum_A_BtC(A, B, C);

  nn::dblvector answer{ 35, 44, 44, 56 };
  const auto& result = A.GetRef();

  EXPECT_EQ(answer.size(), result.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(answer[i], result[i]);
  }
}

TEST(Matrix, accum_y_Atx)
{
  auto A = CreateMatrix(3, 2);
  nn::dblvector x{ 1, 2, 3 };
  nn::dblvector y{ 0.0, 0.0 };
  
  nn::accum_y_Atx(y, A, x);

  nn::dblvector answer{ 22, 28 };
  const auto& result = y;

  EXPECT_EQ(answer.size(), result.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(answer[i], result[i]);
  }
}

TEST(Matrix, accum_y_alphax)
{
  nn::dblvector x{ 1, 2, 3 };
  nn::dblvector y{ 4, 5, 6 };

  nn::dblvector answer{ 4.1, 5.2, 6.3 };

  nn::accum_y_alphax(y, 0.1, x);
  const auto& result = y;

  EXPECT_EQ(answer.size(), result.size());
  for (int i = 0; i < answer.size(); ++i) {
    EXPECT_EQ(answer[i], result[i]);
  }
}