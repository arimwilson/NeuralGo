package neural

import (
  "github.com/gonum/matrix/mat64";
)

func NewLayer(name ActivationName, inputs int, neurons int) *Layer {
  layer := new(Layer)
  layer.Weights = mat64.NewDense(inputs + 1, neurons, nil)
  layer.ActivationFunction = NewActivationFunction(name)
  layer.DActivationFunction = NewDActivationFunction(name)
  return layer
}

type Layer struct {
  Input *mat64.Matrix  // examples x (inputs + 1)
  Weights *mat64.Matrix  // (inputs + 1) x neurons
  ActivationFunction ActivationFunction
  Output *mat64.Matrix  // examples x neurons
  DActivationFunction DActivationFunction
  Gradient *mat64.Matrix  // examples x neurons
}

func (self* Layer) Forward() {
  self.Output.Mult(self.Input, self.Weights)
  self.Output.Apply(
      func (r, c int, v float64) float64 { return self.ActivationFunction(v) },
      self.Output)
}

func (self* Layer) Backward() {
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
}
