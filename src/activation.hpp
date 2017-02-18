#pragma once

#include <algorithm>
#include <numeric>
#include <cmath>

namespace nn
{



class ActivationFunction
{
public:
  virtual void f(dblvec& fx, const dblvec& x) const
  {
    std::transform(std::begin(x), std::end(x), std::begin(fx), [&](auto y){ return this->f_value(y); });
  }

  virtual void df(dblvec& dfx, const dblvec& x, const dblvec& fx) const
  {
    std::transform(std::begin(x), std::end(x),
                   std::begin(fx),
                   std::begin(dfx), [&](auto y, auto z){ return this->df_value(y, z); });
  }
public:
  virtual double f_value(double x) const = 0;
  virtual double df_value(double x, double fx) const = 0;
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
  
private:
  double f_value(double x) const override
  {
    return (gamma/(1 + std::exp(-sigma*x)) - eta);
  }

  double df_value(double x, double fx) const override
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

private:
  double f_value(double x) const override { return x; }
  double df_value(double x, double fx) const override { return 1; }
};



class SoftmaxActivation : public ActivationFunction
{
public:
  virtual void f(dblvec& fx, const dblvec& x) const
  {
    total_value = std::accumulate(std::begin(x), std::end(x), 0.0,
                                  [](auto y, auto z) { return y + std::exp(y); });

    std::transform(std::begin(x), std::end(x), std::begin(fx), [&](auto y){ return this->f_value(y); });
  }

  virtual void df(dblvec& dfx, const dblvec& x, const dblvec& fx) const
  {
    std::transform(std::begin(x), std::end(x),
                   std::begin(fx),
                   std::begin(dfx), [&](auto y, auto z){ return this->df_value(y, z); });
  }

private:
  mutable double total_value;

  double f_value(double x) const override
  {
    return 0.0;
  }

  double df_value(double x, double fx) const override
  {
    return 0;
  }
};



}
