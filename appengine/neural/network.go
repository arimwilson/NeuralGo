package neural

import ("encoding/json"; "github.com/golang/protobuf/proto"; "math/rand")

func NewNetwork(
    networkConfiguration NetworkConfiguration) *Network {
  network := new(Network)
  network.init(networkConfiguration)
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

func (self *Network) Forward(inputs []float64) []float64 {
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

func (self *Network) Backward(values []float64) {
  outputLayer := self.Layers[len(self.Layers) - 1]
  for i, neuron := range outputLayer.Neurons {
    neuron.BackwardOutput(values[i])
  }
  for i := len(self.Layers) - 2; i >= 0; i-- {
    self.Layers[i].Backward()
  }
}

func (self *Network) Update(learningConfiguration LearningConfiguration) {
  for _, layer := range self.Layers {
    layer.Update(*learningConfiguration.Rate)
  }
}

func (self *Network) Train(inputs []float64, values []float64,
                           learning_configuration LearningConfiguration) {
  self.Forward(inputs)
  outputLayer := self.Layers[len(self.Layers) - 1]
  for i, neuron := range outputLayer.Neurons {
    neuron.BackwardOutput(values[i])
  }
  for i := len(self.Layers) - 2; i >= 0; i-- {
    self.Layers[i].Backward()
  }
}

func (self *Network) Serialize() []byte {
  var networkConfiguration NetworkConfiguration
  networkConfiguration.Inputs = proto.Int32(int32(len(self.Inputs)))
  for _, layer := range self.Layers {
    // All neurons in the same layer have the same activation function.
    layerConfiguration := new(LayerConfiguration)
    layerConfiguration.ActivationFunction =
        layer.Neurons[0].ActivationFunction.Enum()
    for _, neuron := range layer.Neurons {
      neuronConfiguration := new(NeuronConfiguration)
      for _, synapse := range neuron.InputSynapses {
        synapseConfiguration := new(SynapseConfiguration)
        synapseConfiguration.Weight = proto.Float64(synapse.Weight)
        neuronConfiguration.Synapse = append(
            neuronConfiguration.Synapse, synapseConfiguration)
      }
      layerConfiguration.Neuron = append(
          layerConfiguration.Neuron, neuronConfiguration)
    }
    networkConfiguration.Layer = append(
        networkConfiguration.Layer, layerConfiguration)
  }
  // TODO(ariw): Return byte representation rather than text representation.
  byteNetwork, _ := json.Marshal(networkConfiguration)
  return byteNetwork
}

func (self *Network) Deserialize(byteNetwork []byte) {
  var networkConfiguration NetworkConfiguration
  json.Unmarshal(byteNetwork, &networkConfiguration)
  self.init(networkConfiguration)
}

func (self *Network) init(networkConfiguration NetworkConfiguration) {
  self.Inputs = []*Input{}
  for i := 0; i < int(*networkConfiguration.Inputs); i++ {
    self.Inputs = append(self.Inputs, NewInput())
  }
  self.Layers = []*Layer{}
  for _, layerConfiguration := range networkConfiguration.Layer {
    layer := NewLayer(
        len(layerConfiguration.Neuron),
        *layerConfiguration.ActivationFunction)
    self.Layers = append(self.Layers, layer)
  }
  // Connect all the layers.
  for _, input := range self.Inputs {
    input.ConnectTo(self.Layers[0])
  }
  for i := 0; i < len(self.Layers) - 1; i++ {
    self.Layers[i].ConnectTo(self.Layers[i+1])
  }
  // Initialize weights if they were specified.
  for i, layerConfiguration := range networkConfiguration.Layer {
    for j, neuronConfiguration := range layerConfiguration.Neuron {
      for k, synapseConfiguration := range neuronConfiguration.Synapse {
        self.Layers[i].Neurons[j].InputSynapses[k].Weight =
            *synapseConfiguration.Weight
      }
    }
  }
}
