#include "network.hpp"
#include "utility.hpp"

#include <algorithm>
#include <functional>

namespace nn
{



Layer::Layer(int size_use, int batch_size_use,
             std::shared_ptr<ActivationFunction> activation_fn_use)
  : size(size_use),
    batch_size(batch_size_use),
    net_input(batch_size, size),
    activation(batch_size, size),
    bias(size),
    activation_fn(activation_fn_use)
{
}



void
Layer::CalculateActivation()
{
  for (int row = 0; row < activation.Rows(); ++row) {
    net_input.SetRowValues(row, bias);
  }

  for (auto& in_conn: incoming) {
    in_conn->AccumulateNetInput(net_input);
  }

  std::transform(begin(net_input), end(net_input), begin(activation),
                 [&](auto& x) { return activation_fn->f(x); });
}



dblscalar
Layer::TotalError(const dblmatrix& target_pattern, const ErrorFunction* error_fn)
{
  return std::inner_product(activation.begin(), activation.end(), target_pattern.begin(), 0.0,
                            std::plus<dblscalar>(),
                            [&](auto x, auto y) { return error_fn->E(x, y); });
}



FeedForwardConnection::FeedForwardConnection(Layer* from, Layer* to)
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


Network::Network(const std::vector<size_t>& layer_sizes,
                 int batch_size_use,
                 std::shared_ptr<ActivationFunction> hid_act_fn,
                 std::shared_ptr<ActivationFunction> out_act_fn,
                 std::shared_ptr<ErrorFunction> err_function_use)
  : batch_size(batch_size_use),
    err_function(err_function_use),
    current_epoch(0),
    last_error(0)
{
  size_t num_hid = layer_sizes.size() - 1;
  
  for (size_t l = 0; l < num_hid; ++l) {
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



dblmatrix
Network::FeedForward(const dblmatrix& input_pattern)
{
  layers[INPUT_LAYER]->SetActivation(input_pattern);

  for (size_t l = 1; l < layers.size(); ++l) {
    layers[l]->CalculateActivation();
  }

  return layers.back()->GetActivation();
}



dblscalar
Network::TotalError(const dblmatrix& target_pattern)
{
  return (last_error = layers.back()->TotalError(target_pattern, err_function.get()));
}


void
Network::AddConnection(Layer* from, Layer *to)
{
  connections.push_back(std::make_shared<FeedForwardConnection>(from, to));
}

void
Network::AddConnection(size_t from, size_t to)
{
  connections.push_back(std::make_shared<FeedForwardConnection>(layers[from].get(), layers[to].get()));
}



}
