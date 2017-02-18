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
    learning_rate(learning_rate_use),
    error_fn(error_fn_use)
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
  // XOR patterns
  //std::vector<dblvec> input{ { 0, 0 },{ 0, 1 },{ 1, 0 },{ 1, 1 } };
  //std::vector<dblvec> target{{-10}, {3}, {1}, {-1.2}};
  //std::vector<dblvec> target{ { -2, -2 },{ -3, 2 },{ 4, -2 },{ 2, 4 } };

  std::vector<dblvec> input{
    { 0, 0, 0, 0},
    { 0, 1, 0, 1},
    { 0, 1, 1, 0},
    { 1, 0, 0, 1}
  };

  std::vector<dblvec> target{
    { -1.75, -1.75, -1.75, -1.75 },
    { -1.75, 0.992, -1.75, 0.992 },
    { -1.75, 0.992, 0.992, -1.75 },
    { 0.992, -1.75, -1.75, 0.992 }
  };

  for (int epoch = 0; epoch < 50'000; ++epoch) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      auto in = input[pattern];
      auto targ = target[pattern];

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

  for (int pattern = 0; pattern < 4; ++pattern) {
    auto in = input[pattern];
    auto targ = target[pattern];

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


}
}
