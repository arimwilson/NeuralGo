package neural_test

import (
  "github.com/gonum/matrix/mat64";
  "github.com/golang/protobuf/proto";
  "testing"
  "../neural";
)

// Example generated from
// http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
func CreateSimpleNetwork(t *testing.T) *neural.Network {
  neuralNetwork := new(neural.Network)
  if err := neuralNetwork.Deserialize([]byte(
      "{\"inputs\":2,\"layer\":[{\"name\":2,\"outputs\":2,\"weight\":[0.15," +
      "0.25,0.2,0.3,0.35,0.35]},{\"name\":2,\"outputs\":2,\"weight\":[0.4," +
      "0.5,0.45,0.55,0.6,0.6]}],\"error_name\":0}")); err != nil {
    t.Fail()
  }
  return neuralNetwork
}

func equalsApprox(a, b, tolerance float64) bool {
  diff := a - b
  return diff < tolerance && -diff < tolerance
}

func TestForward(t *testing.T) {
  neuralNetwork := CreateSimpleNetwork(t)
  inputs := []float64{0.05, 0.10}
  outputs := neuralNetwork.Evaluate(inputs)
  if !equalsApprox(0.75136507, outputs[0], 0.0001) {
    t.Errorf("output 0 %v unexpected", outputs[0])
  }
  if !equalsApprox(0.772928465, outputs[1], 0.0001) {
    t.Errorf("output 1 %v unexpected", outputs[1])
  }
}

func TestBackward(t *testing.T) {
  neuralNetwork := CreateSimpleNetwork(t)
  inputs := mat64.NewDense(1, 2, []float64{0.05, 0.10})
  neuralNetwork.Forward(inputs)
  values := mat64.NewDense(1, 2, []float64{0.01, 0.99})
  neuralNetwork.Backward(values)
  expected_gradient_1 := mat64.NewDense(2, 1, []float64{0.13849856, -0.03809824})
  if !mat64.EqualApprox(
          neuralNetwork.Layers[1].Deltas, expected_gradient_1, 0.0001) {
    t.Errorf("gradient 1 unexpected:\n%v",
             mat64.Formatted(neuralNetwork.Layers[1].Deltas))
  }
  // TODO(ariw): Fill in the other value of layer 0's gradient when known.
  if !equalsApprox(0.00877136, neuralNetwork.Layers[0].Deltas.At(0, 0),
                   0.0001) {
    t.Errorf("gradient 0 unexpected:\n%v",
             mat64.Formatted(neuralNetwork.Layers[0].Deltas))
  }
}

func TestUpdate(t *testing.T) {
  neuralNetwork := CreateSimpleNetwork(t)
  inputs := mat64.NewDense(1, 2, []float64{0.05, 0.10})
  neuralNetwork.Forward(inputs)
  values := mat64.NewDense(1, 2, []float64{0.01, 0.99})
  neuralNetwork.Backward(values)
  learningConfiguration := neural.LearningConfiguration{
      Epochs: proto.Int32(1),
      Rate: proto.Float64(0.5),
      Decay: proto.Float64(0),
      BatchSize: proto.Int32(1),
  }
  neuralNetwork.Update(learningConfiguration)
  expected_weights_0 := mat64.NewDense(
      3, 2, []float64{0.149780716, 0.24975114, 0.19956143, 0.29950229, 0.35,
                      0.35})
  if !mat64.EqualApprox(
          neuralNetwork.Layers[0].Weight, expected_weights_0, 0.0001) {
    t.Errorf("weights 0 unexpected:\n%v",
             mat64.Formatted(neuralNetwork.Layers[0].Weight))
  }
  expected_weights_1 := mat64.NewDense(
      3, 2, []float64{0.35891648, 0.51130127, 0.408666186, 0.561370121, 0.6,
                      0.6})
  if !mat64.EqualApprox(
          neuralNetwork.Layers[1].Weight, expected_weights_1, 0.0001) {
    t.Errorf("weights 1 unexpected:\n%v",
             mat64.Formatted(neuralNetwork.Layers[1].Weight))
  }
}
