#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "error.hpp"

#include <vector>
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

  Layer(int size_use,
        std::shared_ptr<ActivationFunction> activation_fn_use);

  void SetActivationFunction(std::shared_ptr<ActivationFunction> act_fn)
  {
    activation_fn = act_fn;
  }
  
  void SetActivation(const dblvector& in)   { activation = in; } // for input layers
  void CalculateActivation();                                 // for hidden layers

  const dblvector& GetActivation() const { return activation; }
  dblvector& GetActivation() { return activation; }
  const dblvector& GetNetInput() const { return net_input; }

  const double* GetActivationPtr() const { return &activation[0]; }

  int Size() const { return size; }

  auto GetActivationFunction() const { return activation_fn; }

  void AddIncomingConnection(Connection* in)  { incoming.push_back(in); }
  void AddOutgoingConnection(Connection* out) { outgoing.push_back(out); }

private:
  const int size;
  dblvector net_input;
  dblvector activation;
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
      weights(size)
  {
    layer_from->AddOutgoingConnection(this);
    layer_to->AddIncomingConnection(this);
  }

  int Rows() const { return rows; }
  int Cols() const { return cols; }
  int Size() const { return size; }

  void AccumulateNetInput(dblvector& net_input)
  {
    nn::accum_Ax(net_input, 1.0, weights, layer_from->GetActivation(), rows, cols);
  }


  dblvector& GetWeights() { return weights; }

private:
  Layer* layer_from;
  Layer* layer_to;
  
  int    rows;
  int    cols;
  int    size;

  dblvector weights;
};



class Network
{
  friend train::NetworkTrainer;

  static const int INPUT_LAYER = 0;
  
public:
  // Network(const std::string& file_name);        // load a network from a file
  Network(const std::vector<int>& layer_sizes,     // create a network with specified layer sizes
          std::shared_ptr<ActivationFunction> hid_act_fn,
          std::shared_ptr<ActivationFunction> out_act_fn);

  Network() {} // create an empty network
  
  void AddLayer(int size, std::shared_ptr<ActivationFunction> act_fn)
  {
    layers.emplace_back(std::make_shared<Layer>(size, act_fn));
  }

  int AddDefaultConnections();

  dblvector FeedForward(const dblvector& input_pattern);

  const dblvector& GetActivation(int l) const { return layers[l]->GetActivation(); }
  const double* GetActivationPtr(int l) const { return layers[l]->GetActivationPtr(); }

  const dblvector& GetNetInput(int l) const { return layers[l]->GetNetInput(); }

  std::vector<std::shared_ptr<Layer>>& GetLayers() { return layers; }
  std::vector<std::shared_ptr<Connection>>& GetConnections() { return connections; }

private:
  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::shared_ptr<Connection>> connections;

  void AddConnection(Layer* from, Layer* to);
};



}
