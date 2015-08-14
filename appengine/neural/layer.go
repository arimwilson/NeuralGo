package neural

import (
  "fmt";
  "github.com/gonum/matrix/mat64"
)

func NewLayer(name ActivationName, inputs int, outputs int) *Layer {
  layer := new(Layer)
  layer.Weight = mat64.NewDense(inputs + 1, outputs, nil)
  layer.Name = name
  layer.ActivationFunction = NewActivationFunction(layer.Name)
  layer.Output = &mat64.Dense{}
  layer.DActivationFunction = NewDActivationFunction(layer.Name)
  layer.Gradient = &mat64.Dense{}
  return layer
}

type Layer struct {
  Input *mat64.Dense  // examples x (inputs + 1)
  Weight *mat64.Dense  // (inputs + 1) x outputs
  Name ActivationName
  ActivationFunction ActivationFunction
  Output *mat64.Dense  // examples x outputs
  DActivationFunction DActivationFunction
  Gradient *mat64.Dense  // outputs x examples
}

func (self* Layer) Forward(previous *Layer) {
  self.Input = previous.Output
  rows, cols := self.Input.Dims()
  ones := mat64.NewDense(rows, 1, nil)
  for i := 0; i < rows; i++ {
    ones.Set(i, 0, 1.0)
  }
  input_and_bias := &mat64.Dense{}
  input_and_bias.Augment(self.Input, ones)
  rows2, cols2 := self.Weight.Dims()
  fmt.Printf("input: %v %v, weight: %v %v\n", rows, cols + 1, rows2, cols2)
  self.Output.Mul(input_and_bias, self.Weight)
  self.Output.Apply(
      func (r, c int, v float64) float64 { return self.ActivationFunction(v) },
      self.Output)
}

func (self* Layer) Backward(next *Layer) {
  self.Gradient.Mul(self.Weight, next.Gradient)
  self.Gradient.Apply(
      func (r, c int, v float64) float64 { return self.DActivationFunction(v) },
      self.Gradient)
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
  deltas := &mat64.Dense{}
  deltas.Mul(self.Gradient, self.Input)
  deltas = deltas.T().(*mat64.Dense)
  decay := &mat64.Dense{}
  decay.Scale(*learningConfiguration.Decay, self.Weight)
  deltas.Sub(deltas, decay)
  deltas.Scale(*learningConfiguration.Rate, deltas)
  self.Weight.Add(self.Weight, deltas)
}
