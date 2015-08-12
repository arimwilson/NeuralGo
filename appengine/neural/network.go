package neural

import (
  "encoding/json";
  "github.com/golang/protobuf/proto";
  "github.com/gonum/matrix/mat64";
  "math/rand"
)

func NewNetwork(
    networkConfiguration NetworkConfiguration) *Network {
  network := new(Network)
  network.init(networkConfiguration)
  return network
}

type Network struct {
  Layers []*Layer
}

func (self *Network) RandomizeSynapses() {
  for _, layer := range self.Layers {
    rows, cols := layer.Weight.Dims()
    for i := 0; i < rows; i++ {
      for j := 0; j < cols; j++ {
        layer.Weight.Set(i, j, rand.Float64() - 0.5)
      }
    }
  }
}

func (self *Network) Forward(inputs *mat64.Matrix) {
  for _, layer := range self.Layers {
    layer.Forward()
  }
}

func (self *Network) Backward(values *mat64.Matrix) {
  for i := len(self.Layers) - 1; i >= 0; i-- {
    self.Layers[i].Backward()
  }
}

func (self *Network) Update(learningConfiguration LearningConfiguration) {
  for _, layer := range self.Layers {
    layer.Update(learningConfiguration)
  }
}

func (self *Network) Evalaute(features []float64) []float64 {
  inputs := mat64.NewDense(1, len(features), features)
  self.Forward(inputs)
  return self.Layers[len(self.Layers)-1].Outputs.RawRowView(0)
}

func (self *Network) Serialize() []byte {
  var networkConfiguration NetworkConfiguration
  _, inputs := self.Layers[0].Dims()
  networkConfiguration.Inputs = proto.Int32(int32(inputs - 1))
  for _, layer := range self.Layers {
    layerConfiguration := new(LayerConfiguration)
    layerConfiguration.ActivationFunction = layer.ActivationFunction.Enum()
    rows, cols := layer.Dims()
    layerConfiguration.Neurons = proto.Int32(int32(cols))
    for i := 0; i < rows; i++ {
      for j := 0; j < cols; j++ {
        layerConfiguration.Weight = append(
            layerConfiguration.Weight, layer.Weight.At(i, j))
      }
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
  self.Layers = []*Layer{}
  inputs := *networkConfiguration.Inputs
  for i, layerConfiguration := range networkConfiguration.Layer {
    layer := NewLayer(
        inputs, *layerConfiguration.Neurons,
        *layerConfiguration.ActivationFunction)
    self.Layers = append(self.Layers, layer)
    inputs = *layerConfiguration.Neurons
    // Initialize weights if they were specified.
    for i, weight := range layerConfiguration.Weight {
      layer.Input.Set(i / (inputs + 1), i % (inputs + 1), weight)
    }
  }
}
