#pragma once

#include "network.hpp"
#include "input.hpp"

#include <map>

#include <cstring>

namespace nn
{
namespace train
{



// provides access to the internals of the network useful during training
class NetworkTrainer
{
public:
  NetworkTrainer(Network& network_use)
    : network(network_use)
  {
  }

  dblmatrix FeedForward(const dblmatrix& input_pattern)
  {
    return network.FeedForward(input_pattern);
  }

  std::vector<std::shared_ptr<Layer>> GetLayers() const { return network.layers; }
  std::vector<std::shared_ptr<Connection>> GetConnections() const { return network.connections; }

  template <typename PtrType>
  double* GetLayerBiasPtr(PtrType layer) { return &(layer->bias[0]); }
  template <typename PtrType>
  dblvector& GetLayerBias(PtrType layer) { return layer->bias; }
  double* GetLayerActivationPtr(std::shared_ptr<Layer> layer) { return layer->activation.GetPtr(); }
  double* GetLayerNetInputPtr(std::shared_ptr<Layer> layer) { return layer->activation.GetPtr(); }

  //std::vector<Connection *>
  //GetLayerIncomingConnections(std::shared_ptr<Layer> layer) { return layer->incoming; }

  //std::vector<Connection *>
  //GetLayerOutgoingConnections(std::shared_ptr<Layer> layer) { return layer->outgoing; }

  //double* GetConnectionWeightPtr(std::shared_ptr<Connection> c) { return &(c->weights[0]); }

  Layer* GetConnectionFromLayer(std::shared_ptr<Connection> c) { return c->layer_from; }
  Layer* GetConnectionToLayer(std::shared_ptr<Connection> c) { return c->layer_to; }

private:
  Network network;
};




class TrainingAlgorithm
{
public:

  virtual void InitializeNetwork() = 0;
  virtual void Train() = 0;

  virtual ~TrainingAlgorithm() {}
  
private:
  
};




class BackpropLayer;
class BackpropConnection;


class BackpropTrainingAlgorithm : public TrainingAlgorithm
{
public:
  BackpropTrainingAlgorithm(Network& network_use, double learning_rate_use, std::shared_ptr<ErrorFunction> error_fn_use);

  void InitializeNetwork() override;
  void Train() override;

  void SetTrainingData(input::TrainingData* td) { training_data = td; }
  
private:
  NetworkTrainer ntr;

  std::vector<std::shared_ptr<BackpropLayer>> bp_layers;
  std::vector<std::shared_ptr<BackpropConnection>> bp_connections;

  std::shared_ptr<ErrorFunction> error_fn;

  double learning_rate;
  int    max_epochs;

  nn::input::TrainingData* training_data;
};




class BackpropLayer
{
  friend class BackpropTrainingAlgorithm;
  
public:
  BackpropLayer(NetworkTrainer& ntr_use, Layer* layer_use);

  template <typename RngType>
  void InitializeBiases(RngType& randgen)
  {
    auto& bias = ntr.GetLayerBias(layer);

    std::transform(begin(bias), end(bias), begin(bias), std::ref(randgen));
  }

  int Size() const { return layer->Size(); }
  int BatchSize() const { return layer->BatchSize(); }

  void CalculateActivationDerivative()
  {
    auto fn = layer->GetActivationFunction();
    const auto& net_in = layer->GetNetInput();

    std::transform(begin(net_in), end(net_in), begin(activation), begin(activation_df),
                   [&](auto x, auto fx) { return fn->df(x, fx); });
  }

  void CalculateDelta();

  void CalculateDelta(const dblmatrix& target, std::shared_ptr<ErrorFunction> error_fn); // for output layer

  void AddIncomingConnection(BackpropConnection* c) { incoming.push_back(c); }
  void AddOutgoingConnection(BackpropConnection* c) { outgoing.push_back(c); }

  const dblmatrix& GetDelta() const { return delta; }
  const dblmatrix& GetActivation() const { return activation; }

private:
  NetworkTrainer& ntr;
  Layer *layer;
  std::vector<BackpropConnection*> incoming;
  std::vector<BackpropConnection*> outgoing;
  
  dblmatrix& activation;
  dblmatrix activation_df; // derivative of activation function
  dblmatrix delta;
};




struct BackpropTrainingParameters
{
  double learning_rate;
  double momentum;
  bool   normalize_gradient;
};



class BackpropConnection
{
public:
  BackpropConnection(std::shared_ptr<Connection> connection_use,
                     BackpropLayer* from,
                     BackpropLayer* to,
                     double learning_rate_use);

  template <typename RngType>
  void InitializeWeights(RngType& randgen)
  {
    std::transform(begin(weights), end(weights), begin(weights), std::ref(randgen));
  }

  void AccumulateNetDelta(dblmatrix& delta);

  void AccumulateGradients()
  {
    const auto& delta = layer_to->GetDelta();
    const auto& activation = layer_from->GetActivation();
    nn::accum_A_BtC(delta_w, delta, activation);
  }

  void UpdateWeights()
  {
    nn::accum_A_alphaB(delta_w,  params.momentum, delta_w_previous);
    nn::accum_A_alphaB(weights, -params.learning_rate, delta_w);
    delta_w_previous = delta_w;
    std::memset(delta_w.GetPtr(), 0, sizeof(double)*delta_w.Size());
  }

private:
  std::shared_ptr<Connection> connection;
  BackpropLayer* layer_from;
  BackpropLayer* layer_to;
  
  double learning_rate;

  dblmatrix&     weights;
  dblmatrix      delta_w;
  dblmatrix      delta_w_previous;

  BackpropTrainingParameters params;
};





} // namespace train
} // namespace nn
