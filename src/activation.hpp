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

private:
  double slope;

  double f(double x) const override { return slope * x; }
  double df(double x, double fx) const override { return slope; }
};



//class SoftmaxActivation : public ActivationFunction
//{
//public:
//  virtual void f(dblvector& fx, const dblvector& x) const
//  {
//    total_value = std::accumulate(std::begin(x), std::end(x), 0.0,
//                                  [](auto y, auto z) { return y + std::exp(y); });
//
//    std::transform(std::begin(x), std::end(x), std::begin(fx), [&](auto y){ return this->f_value(y); });
//  }
//
//  virtual void df(dblvector& dfx, const dblvector& x, const dblvector& fx) const
//  {
//    std::transform(std::begin(x), std::end(x),
//                   std::begin(fx),
//                   std::begin(dfx), [&](auto y, auto z){ return this->df_value(y, z); });
//  }
//
//private:
//  mutable double total_value;
//
//  double f_value(double x) const override
//  {
//    return 0.0;
//  }
//
//  double df_value(double x, double fx) const override
//  {
//    return 0;
//  }
//};



}
