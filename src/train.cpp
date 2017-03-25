#include "train.hpp"

#include <functional>
#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

namespace nn
{
namespace train
{

BackpropTrainingAlgorithm::BackpropTrainingAlgorithm(Network& network_use,
                                                     const BackpropTrainingParameters params_use)
  : ntr(network_use),
    params(params_use),
    error_fn(ntr.GetErrorFunction()),
    training_data(nullptr)
{
  const auto& layers = ntr.GetLayers();
  const auto& connections = ntr.GetConnections();

  std::map<Layer*, std::shared_ptr<BackpropLayer>> layer_to_bp;

  for (auto& l : layers) {
    auto bp_layer = std::make_shared<BackpropLayer>(ntr, params, l.get(), ntr.GetErrorFunction().get());
    layer_to_bp.insert(std::make_pair(l.get(), bp_layer));
    
    bp_layers.push_back(bp_layer);
  }
  for (auto& c : connections) {
    auto from_layer = ntr.GetConnectionFromLayer(c);
    auto to_layer = ntr.GetConnectionToLayer(c);
    auto bp_from_layer = layer_to_bp[from_layer].get();
    auto bp_to_layer = layer_to_bp[to_layer].get();

    auto bp_connection = std::make_shared<BackpropConnection>(c, bp_from_layer, bp_to_layer, params);
    bp_connections.push_back(bp_connection);
  }
}




void
BackpropTrainingAlgorithm::InitializeNetwork()
{
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 mt_rand(seed);
  auto randgen = std::bind(std::uniform_real_distribution<double>(-0.5, 0.5), mt_rand);

  for (auto& layer : bp_layers) {
    layer->InitializeBiases(randgen);
  }
  for (auto& conn : bp_connections) {
    conn->InitializeWeights(randgen);
  }
}




void
BackpropTrainingAlgorithm::Train()
{
  if (!training_data) {
    std::cerr << "No training data selected." << std::endl;
    return;
  }

  for (int epoch = 0; epoch <= params.max_epochs; ++epoch) {
    ntr.SetCurrentEpoch(epoch);

    dblscalar total_error = 0;

    for (auto& batch = training_data->begin(); batch != training_data->end(); ++batch) {
      const auto& in = batch->Input();
      const auto& targ = batch->Output();

      ntr.FeedForward(in);
      total_error += ntr.TotalError(targ);

      ntr.NotifyBatch();

      // delta at output layer
      auto& output_layer = bp_layers.back();
      output_layer->CalculateActivationDerivative();
      output_layer->CalculateDelta(targ);

      for (int i = bp_layers.size() - 2; i >= 1; --i) {
        bp_layers[i]->CalculateActivationDerivative();
        bp_layers[i]->CalculateDelta();
      }

      for (int i = bp_layers.size() - 1; i >= 1; --i) {
        bp_layers[i]->AccumulateBiasGradient();
        bp_layers[i]->UpdateBias();
      }
    
      for (auto& c : bp_connections) {
        c->AccumulateGradients();
      }
    
      for (auto& c : bp_connections) {
        c->UpdateWeights();
      }
    }

    ntr.NotifyEpoch();

    //std::cout << "epoch " << epoch << '\t' << total_error << std::endl;

    if (total_error < params.min_error) {
      std::cout << epoch << "\t" << total_error << std::endl;
      break;
    }
  }
}


BackpropLayer::BackpropLayer(NetworkTrainer& ntr_use, const BackpropTrainingParameters& params, Layer* layer_use,
  const ErrorFunction* error_fn_use)
  : ntr(ntr_use),
    layer(layer_use),
    learning_rate(params.learning_rate),
    activation(layer->GetActivation()),
    activation_df(layer->BatchSize(), layer->Size()),
    d_bias(layer->Size()),
    delta(layer->BatchSize(), layer->Size()),
    error_fn(error_fn_use)
{
}


template <typename RngType>
void
BackpropLayer::InitializeBiases(RngType& randgen)
{
  auto& bias = ntr.GetLayerBias(layer);

  std::transform(begin(bias), end(bias), begin(bias), std::ref(randgen));
}


void
BackpropLayer::CalculateActivationDerivative()
{
  auto fn = layer->GetActivationFunction();
  auto& net_in = ntr.GetLayerNetInput(layer);

  std::transform(net_in.begin(), net_in.end(), activation.begin(), activation_df.begin(),
                 [&](auto x, auto fx) { return fn->df(x, fx); });
}



void
BackpropLayer::CalculateDelta()
{
  std::fill(begin(delta), end(delta), 0);

  for (auto& conn : outgoing) {
    conn->AccumulateNetDelta(delta);
  }

  // scale by the derivative of the activation
  std::transform(std::begin(delta), std::end(delta), std::begin(activation_df), std::begin(delta),
                 std::multiplies<>());
}



void
BackpropLayer::CalculateDelta(const dblmatrix& target) // for output layer
{
  //CalculateDelta2(target, *layer->GetActivationFunction(), *error_fn);
  //return;

  // // calcualte error into delta
  std::transform(begin(activation), end(activation),
    begin(target),
    begin(delta),
    [&](double x, double y) { return error_fn->dE(x, y); });

  // scale by derivative of activation
  std::transform(begin(delta), end(delta),
    begin(activation_df),
    begin(delta),
    std::multiplies<>());
}


BackpropConnection::BackpropConnection(std::shared_ptr<Connection> connection_use,
                                       BackpropLayer* from,
                                       BackpropLayer* to,
                                       const BackpropTrainingParameters& params_use)
  : connection(connection_use),
    layer_from(from),
    layer_to(to),
    weights(connection->GetWeights()),
    delta_w(layer_to->Size(), layer_from->Size()),
    delta_w_previous(layer_to->Size(), layer_from->Size()),
    params(params_use)
{
  to->AddIncomingConnection(this);
  from->AddOutgoingConnection(this);
}


template <typename RngType>
void
BackpropConnection::InitializeWeights(RngType& randgen)
{
  std::transform(begin(weights), end(weights), begin(weights), std::ref(randgen));

  NguyenWidrowInitialization();
}


void
BackpropConnection::AccumulateNetDelta(dblmatrix& delta)
{
  nn::accum_A_BC(delta, layer_to->GetDelta(), weights);
}


void
BackpropConnection::AccumulateGradients()
{
  const auto& delta = layer_to->GetDelta();
  const auto& activation = layer_from->GetActivation();
  nn::accum_A_BtC(delta_w, delta, activation);
}


void
BackpropConnection::UpdateWeights()
{
  if (params.normalize_gradient) {
    delta_w.Normalize();
  }
  if (params.momentum > 0) {
    nn::accum_A_alphaB(delta_w, params.momentum, delta_w_previous);
    delta_w_previous = delta_w;
  }
  if (params.weight_decay > 0) {
    std::for_each(begin(weights), end(weights), [&](auto& x) { x *= (1 - params.weight_decay); });
  }
  nn::accum_A_alphaB(weights, -params.learning_rate/* / layer_from->BatchSize()*/, delta_w);
  std::memset(delta_w.GetPtr(), 0, sizeof(double)*delta_w.Size());
}


}
}
