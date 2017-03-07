#pragma once

#include "network.hpp"
#include "input.hpp"
#include "utility.hpp"

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

  dblscalar TotalError(const dblmatrix& target_pattern)
  {
    return network.TotalError(target_pattern);
  }


  std::vector<std::shared_ptr<Layer>> GetLayers() const { return network.layers; }
  std::vector<std::shared_ptr<Connection>> GetConnections() const { return network.connections; }

  auto GetErrorFunction() const { return network.err_function; }

  template <typename PtrType>
  double* GetLayerBiasPtr(PtrType layer) { return &(layer->bias[0]); }
  template <typename PtrType>
  dblvector& GetLayerBias(PtrType layer) { return layer->bias; }
  dblscalar* GetLayerActivationPtr(std::shared_ptr<Layer> layer) { return layer->activation.GetPtr(); }
  dblscalar* GetLayerNetInputPtr(std::shared_ptr<Layer> layer) { return layer->activation.GetPtr(); }

  //std::vector<Connection *>
  //GetLayerIncomingConnections(std::shared_ptr<Layer> layer) { return layer->incoming; }

  //std::vector<Connection *>
  //GetLayerOutgoingConnections(std::shared_ptr<Layer> layer) { return layer->outgoing; }

  //double* GetConnectionWeightPtr(std::shared_ptr<Connection> c) { return &(c->weights[0]); }

  Layer* GetConnectionFromLayer(std::shared_ptr<Connection> c) { return c->layer_from; }
  Layer* GetConnectionToLayer(std::shared_ptr<Connection> c) { return c->layer_to; }

private:
  Network& network;
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


struct BackpropTrainingParameters
{
  dblscalar learning_rate;
  dblscalar momentum;
  bool      normalize_gradient;
  // stop when either we hit the maximum number of epochs, or the
  // total error falls below min_error.
  int       max_epochs;
  dblscalar min_error; // NOT USED
};



class BackpropTrainingAlgorithm : public TrainingAlgorithm
{
public:
  BackpropTrainingAlgorithm(Network& network_use, const BackpropTrainingParameters params_use);

  void InitializeNetwork() override;
  void Train() override;

  void SetTrainingData(input::TrainingData* td) { training_data = td; }
  
private:
  NetworkTrainer ntr;

  std::vector<std::shared_ptr<BackpropLayer>> bp_layers;
  std::vector<std::shared_ptr<BackpropConnection>> bp_connections;

  std::shared_ptr<ErrorFunction> error_fn;

  BackpropTrainingParameters params;

  nn::input::TrainingData* training_data;
};




class BackpropLayer
{
  friend class BackpropTrainingAlgorithm;
  
public:
  BackpropLayer(NetworkTrainer& ntr_use, const BackpropTrainingParameters& params, Layer* layer_use, const ErrorFunction* error_fn_use);

  template <typename RngType>
  void InitializeBiases(RngType& randgen)
  {
    auto& bias = ntr.GetLayerBias(layer);

    std::transform(begin(bias), end(bias), begin(bias), std::ref(randgen));
  }

  int Size() const { return layer->Size(); }
  int BatchSize() const { return layer->BatchSize(); }

  template <typename ActivationFunctionType, typename ErrorFunctionType>
  void CalculateDelta(const dblvector& target,
    ActivationFunctionType actf, ErrorFunctionType errf);

  void CalculateActivationDerivative()
  {
    auto fn = layer->GetActivationFunction();
    auto& net_in = layer->GetNetInput();

    std::transform(net_in.begin(), net_in.end(), activation.begin(), activation_df.begin(),
                   [&](auto x, auto fx) { return fn->df(x, fx); });
  }


  void AccumulateBiasGradient()
  {
    dblvector ones(layer->BatchSize(), 1.0);
    accum_y_Atx(d_bias, delta, ones);
  }

  void UpdateBias()
  {
    auto& bias = ntr.GetLayerBias(layer);
    accum_y_alphax(bias, -learning_rate, d_bias);
  }

  void CalculateDelta();  // at hidden layers

  void CalculateDelta(const dblmatrix& target); // for output layer

  void AddIncomingConnection(BackpropConnection* c) { incoming.push_back(c); }
  void AddOutgoingConnection(BackpropConnection* c) { outgoing.push_back(c); }

  const dblmatrix& GetDelta() const { return delta; }
  const dblmatrix& GetActivation() const { return activation; }

private:
  NetworkTrainer& ntr;
  Layer *layer;
  std::vector<BackpropConnection*> incoming;
  std::vector<BackpropConnection*> outgoing;
  
  double learning_rate;

  dblmatrix& activation;
  dblmatrix activation_df; // derivative of activation function
  dblmatrix delta;

  dblvector d_bias;        // delta for bias

  const ErrorFunction* error_fn;
};




class BackpropConnection
{
public:
  BackpropConnection(std::shared_ptr<Connection> connection_use,
                     BackpropLayer* from,
                     BackpropLayer* to,
                     const BackpropTrainingParameters& params);

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
    if (params.normalize_gradient) {
      delta_w.Normalize();
    }
    nn::accum_A_alphaB(delta_w,  params.momentum, delta_w_previous);
    nn::accum_A_alphaB(weights, -params.learning_rate, delta_w);
    delta_w_previous = delta_w;
    std::memset(delta_w.GetPtr(), 0, sizeof(double)*delta_w.Size());
  }

private:
  std::shared_ptr<Connection> connection;
  BackpropLayer* layer_from;
  BackpropLayer* layer_to;
  
  dblmatrix&     weights;
  dblmatrix      delta_w;
  dblmatrix      delta_w_previous;

  BackpropTrainingParameters params;
};





} // namespace train
} // namespace nn
