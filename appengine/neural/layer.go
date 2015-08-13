package neural

import (
  "github.com/gonum/matrix/mat64";
)

func NewLayer(name ActivationName, inputs int, outputs int) *Layer {
  layer := new(Layer)
  layer.Weight = mat64.NewDense(inputs + 1, outputs, nil)
  layer.Name = name
  layer.ActivationFunction = NewActivationFunction(layer.Name)
  layer.DActivationFunction = NewDActivationFunction(layer.Name)
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
  self.Input.Augment(self.Input, ones)
  self.Output.Mul(self.Input, self.Weight)
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
  rows, cols := self.Gradient.Dims()
  deltas := mat64.NewDense(rows, cols, nil)
  deltas.Mul(self.Gradient, self.Input)
  deltas = deltas.T()
  rows, cols = self.Weight.Dims()
  decay := mat64.NewDense(rows, cols, nil)
  decay.Scale(*learningConfiguration.Decay, self.Weight)
  deltas.Sub(deltas, decay)
  deltas.Scale(*learningConfiguration.Rate, deltas)
  self.Weight.Add(self.Weight, deltas)
}
