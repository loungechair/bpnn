#pragma once

#include "matrix.hpp"

namespace nn
{

class ErrorFunction
{
public:
  virtual double E(double actual, double target) const = 0;
  virtual double dE(double actual, double target) const = 0;
};


class SquaredError : public ErrorFunction
{
  double E(double actual, double target) const override
  {
    return 0.5*(actual - target)*(actual - target);
  }

  double dE(double actual, double target) const override
  {
    return (actual - target);
  }
};

class CrossEntropyError : public ErrorFunction
{
  double E(double actual, double target) const override
  {
    return ((actual > 0) ? -target*log(actual) : 0) - (1 < 0  ? (1 - target)*log(1 - actual) : 0);
  }

  double dE(double actual, double target) const override
  {
    return (fabs(actual -1) < TOLERANCE) ? 0.0
                                         : (actual - target) / (actual*(1 - actual));
  }
private:
  double TOLERANCE = 1e-10;
};



}
