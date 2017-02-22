#include "train.hpp"

#include <functional>
#include <algorithm>
#include <chrono>
#include <random>

namespace nn
{
namespace train
{


BackpropTrainingAlgorithm::BackpropTrainingAlgorithm(Network& network_use,
                                                     double learning_rate_use,
                                                     std::shared_ptr<ErrorFunction> error_fn_use)
  : ntr(network_use),
    error_fn(error_fn_use),
    learning_rate(learning_rate_use),
    max_epochs(150'000),
    training_data(nullptr)
{
  const auto& x = ntr.GetLayers();
  const auto& y = ntr.GetConnections();

  std::map<Layer*, std::shared_ptr<BackpropLayer>> layer_to_bp;

  for (auto& l : x) {
    auto bp_layer = std::make_shared<BackpropLayer>(ntr, l.get());
    layer_to_bp.insert(std::make_pair(l.get(), bp_layer));
    
    bp_layers.push_back(bp_layer);
  }
  for (auto& c : y) {
    auto from_layer = ntr.GetConnectionFromLayer(c);
    auto to_layer = ntr.GetConnectionToLayer(c);
    auto bp_from_layer = layer_to_bp[from_layer].get();
    auto bp_to_layer = layer_to_bp[to_layer].get();
    auto bp_connection = std::make_shared<BackpropConnection>(c, bp_from_layer, bp_to_layer, learning_rate);
    bp_connections.push_back(bp_connection);
  }
}




void
BackpropTrainingAlgorithm::InitializeNetwork()
{
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 mt_rand(seed);
  auto randgen = std::bind(std::uniform_real_distribution<double>(-0.2, 0.2), mt_rand);

  int i = 0;
  for (auto& layer : bp_layers) {
    layer->InitializeBiases(randgen);
  }

  i = 0;
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
  for (int epoch = 0; epoch < max_epochs; ++epoch) {
    for (int pattern = 0; pattern < training_data->in.size(); ++pattern) {
      auto in = training_data->in[pattern];
      auto targ = training_data->out[pattern];

      const auto& output = ntr.FeedForward(in);

      // delta at output layer
      auto& output_layer = bp_layers.back();
      output_layer->CalculateActivationDerivative();
      output_layer->CalculateDelta(targ, error_fn);

      for (int i = bp_layers.size() - 2; i >= 1; --i) {
        bp_layers[i]->CalculateActivationDerivative();
        bp_layers[i]->CalculateDelta();
      }

      for (auto& c : bp_connections) {
        c->AccumulateGradients();
      }
    }
    for (auto& c : bp_connections) {
      c->UpdateWeights();
    }
  }

  for (int pattern = 0; pattern < training_data->in.size(); ++pattern) {
    auto in = training_data->in[pattern];
    auto targ = training_data->out[pattern];

    const auto& output = ntr.FeedForward(in);

    auto& output_layer = bp_layers.back();
    std::cout << "{";
    for (auto& x : in) {
      std::cout << "\t" << x;
    }
    std::cout << "} --> {";
    for (auto& x : output_layer->GetActivation()) {
      std::cout << "\t" << x;
    }
    std::cout << "}" << std::endl;
  }
}


BackpropLayer::BackpropLayer(NetworkTrainer& ntr_use, Layer* layer_use)
  : ntr(ntr_use),
    layer(layer_use),
    activation(layer->GetActivation()),
    activation_df(layer->Size()),
    delta(layer->Size())
{
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
BackpropLayer::CalculateDelta(const dblvector& target, std::shared_ptr<ErrorFunction> error_fn) // for output layer
{
  // // calcualte error into delta
  std::transform(begin(activation), end(activation),
    begin(target),
    begin(delta),
    [&](double x, double y) { return error_fn->dE(x, y); });

  // scale by derivative of activation
  std::transform(begin(delta), end(delta),
    begin(activation_df),
    begin(delta),
    std::multiplies<double>());
}

BackpropConnection::BackpropConnection(std::shared_ptr<Connection> connection_use,
                                       BackpropLayer* from,
                                       BackpropLayer* to,
                                       double learning_rate_use)
  : connection(connection_use),
    layer_from(from),
    layer_to(to),
    learning_rate(learning_rate_use),
    weights(connection->GetWeights()),
    delta_w(connection->Size(), 0),
    delta_w_previous(connection->Size(), 0),
    params{ .01, 0.9, false }
{
  to->AddIncomingConnection(this);
  from->AddOutgoingConnection(this);
}


void
BackpropConnection::AccumulateNetDelta(dblvector& delta)
{
  nn::accum_ATx(&delta[0], 1.0, &weights[0],
                        &(layer_to->GetDelta()[0]),
                        connection->Rows(), connection->Cols());
}



}
}
