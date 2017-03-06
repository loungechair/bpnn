#pragma once

#include <algorithm>
#include <numeric>
#include <cmath>

namespace nn
{



class ActivationFunction
{
public:
  virtual double f(double x) const = 0;
  virtual double df(double x, double fx) const = 0;
};



class SigmoidActivation : public ActivationFunction
{
public:
  SigmoidActivation(double min_val, double max_val, double slope = 1.0)
    : gamma(max_val - min_val),
      eta(-min_val),
      sigma(slope),
      sigma_over_gamma(sigma/gamma)
  {}
  
  double f(double x) const override
  {
    return (gamma/(1 + std::exp(-sigma*x)) - eta);
  }

  double df(double x, double fx) const override
  {
    return sigma_over_gamma*(eta + fx)*(gamma - eta - fx);
  }

private:
  double gamma;
  double eta;
  double sigma;
  const double sigma_over_gamma;
};



class LinearActivation : public ActivationFunction
{
public:
  LinearActivation(double slope_use = 1.0) : slope(slope_use) {}

  double f(double x) const override { return slope * x; }
  double df(double x, double fx) const override { return slope; }

private:
  double slope;
};



class TanhActivation : public ActivationFunction
{
public:
  double f(double x) const override
  {
    return tanh(x);
  }

  double df(double x, double fx) const override
  {
    return (1 - fx*fx);
  }
};


}
