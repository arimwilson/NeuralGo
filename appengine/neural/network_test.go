package neural_test

import (
  "fmt";
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
      "0.5,0.45,0.55,0.6,0.6]}]}")); err != nil {
    t.Fail()
  }
  return neuralNetwork
}

func approximatelyEqual(a, b, tolerance float64) bool {
  diff := a - b
  return diff < tolerance && -diff < tolerance
}

func TestForward(t *testing.T) {
  neuralNetwork := CreateSimpleNetwork(t)
  inputs := []float64{0.05, 0.10}
  outputs := neuralNetwork.Evaluate(inputs)
  if !approximatelyEqual(0.75136507, outputs[0], 0.001) {
    t.Errorf("output %v unexpected", outputs[0])
  }
  if !approximatelyEqual(0.772928465, outputs[1], 0.001) {
    t.Errorf("output %v unexpected", outputs[1])
  }
}

func TestBackward(t *testing.T) {
  neuralNetwork := CreateSimpleNetwork(t)
  inputs := mat64.NewDense(1, 2, []float64{0.05, 0.10})
  neuralNetwork.Forward(inputs)
  values := mat64.NewDense(1, 2, []float64{0.01, 0.99})
  neuralNetwork.Backward(values)
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
}
