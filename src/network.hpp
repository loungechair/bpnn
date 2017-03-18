#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "error.hpp"
#include "utility.hpp"

#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include <iostream>
#include <iomanip>

#include <cassert>


namespace nn
{
class Layer;
class Connection;
class Network;


namespace train
{
class NetworkTrainer;
}



class Layer
{
  friend train::NetworkTrainer;
  
public:

  Layer(int size_use, int batch_size_use, std::shared_ptr<ActivationFunction> activation_fn_use);

  void SetActivationFunction(std::shared_ptr<ActivationFunction> act_fn) { activation_fn = act_fn; }
  
  void SetActivation(const dblmatrix& in) { activation = in; } // for input layers
  void CalculateActivation();                                  // for hidden layers

  int BatchSize() const { return batch_size; }

  const dblmatrix& GetActivation() const { return activation; }
  dblmatrix& GetActivation() { return activation; }

  dblscalar TotalError(const dblmatrix& target_pattern, const ErrorFunction* error_fn);

  int Size() const { return size; }

  auto GetActivationFunction() const { return activation_fn; }

  void AddIncomingConnection(Connection* in)  { incoming.push_back(in); }
  void AddOutgoingConnection(Connection* out) { outgoing.push_back(out); }

private:
  const int size;
  int batch_size;
  dblmatrix net_input;
  dblmatrix activation;
  dblvector bias;

  std::shared_ptr<ActivationFunction> activation_fn;

  std::vector<Connection *> incoming;
  std::vector<Connection *> outgoing;
};




class Connection
{
  friend train::NetworkTrainer;

public:
  Connection(Layer* from, Layer* to);

  int Rows() const { return rows; }
  int Cols() const { return cols; }
  int Size() const { return size; }

  void AccumulateNetInput(dblmatrix& net_input)
  {
    nn::accum_A_BCt(net_input, layer_from->GetActivation(), weights);
  }

  dblmatrix& GetWeights() { return weights; }

private:
  Layer* layer_from;
  Layer* layer_to;
  
  int    rows;
  int    cols;
  int    size;

  dblmatrix weights;
};



class Network : public utility::Observable
{
  friend train::NetworkTrainer;

  static const int INPUT_LAYER = 0;
  
public:
  // Network(const std::string& file_name);        // load a network from a file

  Network(const std::vector<int>& layer_sizes,     // create a network with specified layer sizes
          int batch_size_use,
          std::shared_ptr<ActivationFunction> hid_act_fn,
          std::shared_ptr<ActivationFunction> out_act_fn,
          std::shared_ptr<ErrorFunction> err_function_use);

  Network() {} // create an empty network
  
  void AddLayer(int size, std::shared_ptr<ActivationFunction> act_fn)
  {
    layers.emplace_back(std::make_shared<Layer>(size, batch_size, act_fn));
  }

  int AddDefaultConnections();

  dblmatrix FeedForward(const dblmatrix& input_pattern);
  dblscalar TotalError(const dblmatrix& target_pattern);

  int GetCurrentEpoch() const { return current_epoch; }
  double GetLastError() const { return last_error; }

private:
  int batch_size;
  int current_epoch;

  double last_error; // total error across all patterns from last call to FeedForward

  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::shared_ptr<Connection>> connections;

  std::shared_ptr<ErrorFunction> err_function;

  void AddConnection(Layer* from, Layer* to);
};


class SimpleRecurrentNetwork
{
public:

private:
  Network ff_network;
};


template <typename T>
class ErrorStatistics : public utility::Observer
{
public:
  ErrorStatistics(int save_freq_use, const Network& network_use)
    : save_frequency(save_freq_use),
    network(network_use)
  {}

  void UpdateBatch() override
  {
    int epoch = network.GetCurrentEpoch();
    AccumulateErrorStatistics(epoch);
  }

  void UpdateEpoch() override {}

  void AccumulateErrorStatistics(int epoch)
  {
    if ((epoch == 0) || (epoch % save_frequency == 0)) {
      total_error[epoch] += network.GetLastError();
    }
  }

  T GetTotalError(int epoch) const { return total_error[epoch]; }

  auto begin() { return total_error.begin(); }
  auto end() { return total_error.end(); }
  auto begin() const { return total_error.begin(); }
  auto end() const { return total_error.end(); }

private:
  const Network& network;
  int save_frequency; // how many epochs between saving data.  set to 1 to save all the time
  std::map<int, T> total_error;
};


class ErrorPrinter : public utility::Observer
{
public:
  ErrorPrinter(int save_freq_use, const Network& network_use, const utility::Timer* timer_use = nullptr)
    : save_frequency(save_freq_use),
      total_error(0.0),
      network(network_use),
      timer(timer_use)
  {}

  void UpdateBatch() override
  {
    total_error += network.GetLastError();
  }

  void UpdateEpoch() override
  {
    int epoch = network.GetCurrentEpoch();
    if ((epoch == 0) || (epoch % save_frequency == 0)) {
      std::cout << std::setw(6) << epoch << ' ';
      std::cout << std::setw(17) << std::setprecision(4) << std::fixed << total_error;
      if (timer) {
        std::cout << "   " << timer->GetElapsedTimeAsString();
      }
      std::cout << std::endl;
    }
    total_error = 0.0;
  }

private:
  int save_frequency; // how many epochs between saving data.  set to 1 to save all the time
  double total_error;
  const Network& network;
  const utility::Timer* timer;
};


}
