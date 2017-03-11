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

#include <cassert>


namespace nn
{
// forward declarations
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

  Layer(int size_use, int batch_size_use,
        std::shared_ptr<ActivationFunction> activation_fn_use);

  void SetActivationFunction(std::shared_ptr<ActivationFunction> act_fn)
  {
    activation_fn = act_fn;
  }
  
  void SetActivation(const dblmatrix& in)   { activation = in; } // for input layers
  void CalculateActivation();                                 // for hidden layers

  int BatchSize() const { return batch_size; }

  const dblmatrix& GetActivation() const { return activation; }
  dblmatrix& GetActivation() { return activation; }
  const dblmatrix& GetNetInput() const { return net_input; }

  const dblscalar* GetActivationPtr() const { return activation.GetPtr(); }

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
  Connection(Layer* from, Layer* to)
    : layer_from(from),
      layer_to(to),
      rows(layer_to->Size()),
      cols(layer_from->Size()),
      size(rows*cols),
      weights(rows, cols)
  {
    layer_from->AddOutgoingConnection(this);
    layer_to->AddIncomingConnection(this);
  }

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

  std::vector<std::shared_ptr<Layer>>& GetLayers() { return layers; }
  std::vector<std::shared_ptr<Connection>>& GetConnections() { return connections; }

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



template <typename T>
class ErrorStatistics : public utility::Observer
{
  struct ErrorData
  {
    int epoch;
    int num_patterns;
    T total_error;
    T mean;
    T M2;
  };

public:
  ErrorStatistics(int save_freq_use, const Network& network_use)
    : save_frequency(save_freq_use),
    network(network_use)
  {}

  void Update() override
  {
    int epoch = network.GetCurrentEpoch();
    AccumulateErrorStatistics(epoch);
  }

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
  ErrorPrinter(int save_freq_use, const Network& network_use)
    : save_frequency(save_freq_use),
    network(network_use)
  {}

  void Update() override
  {
    int epoch = network.GetCurrentEpoch();
    if ((epoch == 0) || (epoch % save_frequency == 0)) {
      std::cout << epoch << "\t" << network.GetLastError() << std::endl;
    }
  }

private:
  const Network& network;
  int save_frequency; // how many epochs between saving data.  set to 1 to save all the time
};


}
