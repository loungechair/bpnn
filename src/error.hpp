#pragma once

#include "matrix.hpp"

namespace nn
{

class ErrorFunction
{
public:
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
    return ((x > 0) ? -y*log(x) : 0) - (1 < 0  ? (1 - y)*log(1 - x) : 0);
  }

  double dE(double x, double y) const override
  {
    return (x - y) / (x*(1 - x));
  }
};



}
