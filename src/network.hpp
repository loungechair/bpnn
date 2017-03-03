#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "error.hpp"

#include <vector>
#include <memory>
#include <algorithm>

#include <iostream>

#include <cassert>


/*
 * Error statistics
 * - Total error
 * - Average error per pattern
 * - number of patterns
 * - total error by unit
 * - total error by epoch
 */


namespace nn
{
  class Network;

  class ErrorStatistics
  {
  public:
    ErrorStatistics(int num_patterns_use, int num_units_use);

    void GatherStatistics(const Network& net);


    void GetTotalError(int epoch);
    void GetTotalError(int epoch, int pattern);
    void GetTotalError(int epoch, int pattern, int unit);
    //void GetTotalError(int epoch, int unit);

  private:
  };


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



class Network
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

  const dblmatrix& GetActivation(int l) const { return layers[l]->GetActivation(); }
  const dblscalar* GetActivationPtr(int l) const { return layers[l]->GetActivationPtr(); }

  const dblmatrix& GetNetInput(int l) const { return layers[l]->GetNetInput(); }

  std::vector<std::shared_ptr<Layer>>& GetLayers() { return layers; }
  std::vector<std::shared_ptr<Connection>>& GetConnections() { return connections; }

private:
  int batch_size;

  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::shared_ptr<Connection>> connections;

  std::shared_ptr<ErrorFunction> err_function;

  void AddConnection(Layer* from, Layer* to);
};



}
