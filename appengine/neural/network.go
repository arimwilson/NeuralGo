package neural

import (
  "encoding/json";
  "bytes";
  "fmt";
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

func (self *Network) Forward(inputs *mat64.Dense) {
  previous := &Layer{}
  previous.Output = inputs
  for _, layer := range self.Layers {
    layer.Forward(previous)
    previous = layer
  }
}

func (self *Network) Backward(values *mat64.Dense) {
  next := self.Layers[len(self.Layers) - 1]
  next.BackwardOutput(values)
  for i := len(self.Layers) - 2; i >= 0; i-- {
    self.Layers[i].Backward(next)
    next = self.Layers[i]
  }
}

func (self *Network) Update(learningConfiguration LearningConfiguration) {
  for _, layer := range self.Layers {
    layer.Update(learningConfiguration)
  }
}

func (self *Network) Evaluate(features []float64) []float64 {
  inputs := mat64.NewDense(1, len(features), features)
  self.Forward(inputs)
  return self.Layers[len(self.Layers)-1].Output.RawRowView(0)
}

func (self *Network) Serialize() []byte {
  var networkConfiguration NetworkConfiguration
  _, inputs := self.Layers[0].Weight.Dims()
  networkConfiguration.Inputs = proto.Int32(int32(inputs - 1))
  for _, layer := range self.Layers {
    layerConfiguration := new(LayerConfiguration)
    *layerConfiguration.Name = layer.Name
    rows, cols := layer.Weight.Dims()
    layerConfiguration.Outputs = proto.Int32(int32(cols))
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

func (self *Network) Deserialize(byteNetwork []byte) error {
  var networkConfiguration NetworkConfiguration
  if err := json.Unmarshal(byteNetwork, &networkConfiguration); err != nil {
    return err
  }
  self.init(networkConfiguration)
  return nil
}

func (self *Network) DebugString() string {
  var buffer bytes.Buffer
  for i, layer := range self.Layers {
    buffer.WriteString(fmt.Sprintf("layer %v:\n%v", i, layer.DebugString()))
  }
  return buffer.String()
}

func (self *Network) init(networkConfiguration NetworkConfiguration) {
  self.Layers = []*Layer{}
  inputs := int(*networkConfiguration.Inputs)
  for _, layerConfiguration := range networkConfiguration.Layer {
    outputs := int(*layerConfiguration.Outputs)
    layer := NewLayer(*layerConfiguration.Name, inputs, outputs,
                      layerConfiguration.Weight)
    self.Layers = append(self.Layers, layer)
    inputs = outputs
  }
}
