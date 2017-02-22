#include "network.hpp"
#include "utility.hpp"

#include <algorithm>
#include <functional>

namespace nn
{



Layer::Layer(int size_use,
             std::shared_ptr<ActivationFunction> activation_fn_use)
  : size(size_use),
    net_input(size),
    activation(size),
    bias(size),
    activation_fn(activation_fn_use)
{
}



void
Layer::CalculateActivation()
{
  net_input = bias;

  for (auto& in_conn: incoming) {
    in_conn->AccumulateNetInput(net_input);
  }

  activation_fn->f(activation, net_input);
}



Network::Network(const std::vector<int>& layer_sizes,
                 std::shared_ptr<ActivationFunction> hid_act_fn,
                 std::shared_ptr<ActivationFunction> out_act_fn)
{
  int num_hid = layer_sizes.size() - 1;
  
  for (int l = 0; l < num_hid; ++l) {
    AddLayer(layer_sizes[l], hid_act_fn);
  }
  AddLayer(layer_sizes[num_hid], out_act_fn);
  AddDefaultConnections();
}



int
Network::AddDefaultConnections()
{
  nn::utility::adjacent_pairs(std::begin(layers), std::end(layers),
                              [&](auto p, auto q) { this->AddConnection(p.get(), q.get()); });

  return connections.size();
}



dblvector
Network::FeedForward(const dblvector& input_pattern)
{
  layers[INPUT_LAYER]->SetActivation(input_pattern);

  // std::cout << "FORWARD PHASE" << std::endl;
  
  for (size_t l = 1; l < layers.size(); ++l) {
    layers[l]->CalculateActivation();
    
    //std::cout << "Layer " << l << ":";
    //for (auto& p : layers[l]->GetActivation()) {
    //  std::cout << '\t' << p;
    //}
    //
    //std::cout << std::endl;
  }

  return layers.back()->GetActivation();
}



void
Network::AddConnection(Layer* from, Layer *to)
{
  connections.push_back(std::make_shared<Connection>(from, to));
}



}
