#pragma once

#include "matrix.hpp"

namespace nn
{

class ErrorFunction
{
public:
  // virtual void E(dblvector& out, const dblvector& x, const dblvector& target) = 0;
  // virtual void dE(dblvector& out, const dblvector& x, const dblvector& target) = 0;
  virtual double E(double x, double y) const = 0;
  virtual double dE(double x, double y) const = 0;
};


class SquaredError : public ErrorFunction
{
  double E(double x, double y) const override
  {
    return 0.5*(x - y)*(x - y);
  }

  double dE(double x, double y) const override
  {
    return (x - y);
  }
};



}
