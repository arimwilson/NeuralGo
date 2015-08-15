package neural

import (
  "fmt";
  "github.com/gonum/matrix/mat64"
)

func NewLayer(name ActivationName, inputs int, outputs int,
              weight []float64) *Layer {
  layer := new(Layer)
  layer.Weight = mat64.NewDense(inputs + 1, outputs, weight)
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
  rows, _ := self.Input.Dims()
  ones := mat64.NewDense(rows, 1, nil)
  for i := 0; i < rows; i++ {
    ones.Set(i, 0, 1.0)
  }
  input_and_bias := &mat64.Dense{}
  input_and_bias.Augment(self.Input, ones)
  self.Output.Mul(input_and_bias, self.Weight)
  self.Output.Apply(
      func (r, c int, v float64) float64 { return self.ActivationFunction(v) },
      self.Output)
}

func (self* Layer) Backward(next *Layer) {
  fmt.Printf("weight: %v\n, gradient: %v\n", mat64.Formatted(self.Weight), mat64.Formatted(next.Gradient))
  rows, cols := next.Gradient.Dims()
  self.Gradient.Mul(self.Weight, next.Gradient.View(0, 0, rows - 1, cols))
  self.Gradient.Apply(
      func (r, c int, v float64) float64 { return self.DActivationFunction(v) },
      self.Gradient)
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
  deltas := &mat64.Dense{}
  deltas.Mul(self.Gradient, self.Input)
  deltas.TCopy(deltas)
  // decay := &mat64.Dense{}
  // decay.Scale(*learningConfiguration.Decay, self.Weight)
  // deltas.Sub(deltas, decay)
  deltas.Scale(*learningConfiguration.Rate, deltas)
  self.Weight.Add(self.Weight, deltas)
}
