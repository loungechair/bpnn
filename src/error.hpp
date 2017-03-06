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

class CrossEntropyError : public ErrorFunction
{
  double E(double x, double y) const override
  {
    return -y*log(x) - (1 - y)*log(1 - x);
  }

  double dE(double x, double y) const override
  {
    return (x - y) / (x*(1 - x));
  }
};



}
