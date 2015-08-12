package neural

import (
  "github.com/gonum/matrix/mat64";
)

func NewLayer(name ActivationName, inputs int, neurons int) *Layer {
  layer := new(Layer)
  layer.Weight = mat64.NewDense(inputs + 1, neurons, nil)
  layer.ActivationFunction = NewActivationFunction(name)
  layer.DActivationFunction = NewDActivationFunction(name)
  return layer
}

type Layer struct {
  Input *mat64.Matrix  // examples x (inputs + 1)
  Weight *mat64.Matrix  // (inputs + 1) x outputs
  ActivationFunction ActivationFunction
  Output *mat64.Matrix  // examples x outputs
  DActivationFunction DActivationFunction
  Gradient *mat64.Matrix  // outputs x examples
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
  rows, cols := self.Weight.Dims()
  decay := self.mat64.NewDense(rows, cols, nil)
  decay.Scale(*learningConfiguration.Decay, self.Weight)
  rows, cols = self.Gradient.Dims()
  deltas := mat64.NewDense(rows, cols, nil)
  deltas.Scale(*learningConfiguration.Rate,
               deltas.Sub(deltas.Mul(self.Gradient, self.Input), decay))
  self.Weight.Add(self.Weight, deltas)
}
