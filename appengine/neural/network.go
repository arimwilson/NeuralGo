package neural

import ("encoding/json"; "github.com/golang/protobuf/proto"; "math/rand")

func NewNetwork(
    network_configuration NetworkConfiguration) *Network {
  network := new(Network)
  network.Inputs = []*Input{}
  for i := 0; i < network_configuration.Inputs; i++ {
    network.Inputs = append(network.Inputs, NewInput())
  }
  network.Layers = []*Layer{}
  for _, layer_configuration := range network_configuration.Layer {
    layer := NewLayer(count, layer_configuration.ActivationFunction)
    network.Layers = append(network.Layers, layer)
  }
  // Connect all the layers.
  for _, input := range network.Inputs {
    input.ConnectTo(network.Layers[0])
  }
  for i := 0; i < len(network.Layers) - 1; i++ {
    network.Layers[i].ConnectTo(network.Layers[i+1])
  }
  // Initialize weights.
  for i, layer_configuration := range network_configuration.Layers {
    for j, neuron_configuration := range layer_configuration.Neuron {
      for k, synapse_configuration := range neuron_configuration.Synapse {
        network.Layers[i].Neurons[j].InputSynapses[k].Weight =
            synapse_configuration.Weight
      }
    }
  }
  return network
}

type Network struct {
  Inputs []*Input
  Layers []*Layer
}

func (self *Network) RandomizeSynapses() {
  for _, layer := range self.Layers {
    for _, neuron := range layer.Neurons {
      for _, synapse := range neuron.InputSynapses {
        synapse.Weight = rand.Float64() - 0.5
      }
    }
  }
}

func (self *Network) Evaluate(inputs []float64) []float64 {
  for i, input := range inputs {
    self.Inputs[i].Input = input
    self.Inputs[i].Forward()
  }
  for _, layer := range self.Layers {
    layer.Forward()
  }
  outputLayer := self.Layers[len(self.Layers) - 1]
  outputs := make([]float64, len(outputLayer.Neurons))
  for i, neuron := range(outputLayer.Neurons) {
    outputs[i] = neuron.Output
  }
  return outputs
}

func (self *Network) Train(inputs []float64, values []float64, speed float64) {
  self.Evaluate(inputs)
  outputLayer := self.Layers[len(self.Layers) - 1]
  for i, neuron := range outputLayer.Neurons {
    neuron.BackwardOutput(values[i])
  }
  for i := len(self.Layers) - 2; i >= 0; i-- {
    self.Layers[i].Backward()
  }
  for _, layer := range self.Layers {
    layer.Update(speed)
  }
}

func (self *Network) Serialize() []byte {
  serializedNetwork := &SerializedNetwork{
      Inputs: len(self.Inputs),
      Weights: make([][][]float64, len(self.Layers))}
  for i, layer := range self.Layers {
    // All neurons in the same layer have the same activation function.
    serializedNetwork.ActivationFunctions = append(
        serializedNetwork.ActivationFunctions,
        layer.Neurons[0].ActivationFunction)
    serializedNetwork.Weights[i] = make([][]float64, len(layer.Neurons))
    for j, neuron := range layer.Neurons {
      serializedNetwork.Weights[i][j] = make(
          []float64, len(neuron.InputSynapses))
      for k, synapse := range neuron.InputSynapses {
        serializedNetwork.Weights[i][j][k] = synapse.Weight
      }
    }
  }
  byteNetwork, _ := json.Marshal(serializedNetwork)
  return byteNetwork
}

func (self *Network) Deserialize(byteNetwork []byte) {
  serializedNetwork := &SerializedNetwork{}
  json.Unmarshal(byteNetwork, serializedNetwork)
  layers := make([]int, len(serializedNetwork.Weights))
  for i, layer := range serializedNetwork.Weights {
    layers[i] =len(layer)
  }
  self.init(serializedNetwork.Inputs, layers,
            serializedNetwork.ActivationFunctions)
  // Now initialize all the weights.
  for i, layer := range self.Layers {
    for j, neuron := range layer.Neurons {
      for k, synapse := range neuron.InputSynapses {
        synapse.Weight = serializedNetwork.Weights[i][j][k]
      }
    }
  }
}

