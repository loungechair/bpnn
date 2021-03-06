#pragma once

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
    //if (fabs(actual - target) < 0.2) { return 0; }
    return ((actual > 0) ? -target*log(actual) : 0) - (actual < 1  ? (1 - target)*log(1 - actual) : 0);
  }

  double dE(double actual, double target) const override
  {
    //if (fabs(actual - target) < 0.2) { return 0; }
    return (fabs(actual - 1) < TOLERANCE) ? 0.0
                                          : (actual - target) / (actual*(1 - actual));
  }
private:
  double TOLERANCE = 1e-10;
};



}
