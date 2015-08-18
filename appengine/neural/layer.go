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
  // Add bias to input.
  inputAndBias := &mat64.Dense{}
  inputAndBias.Augment(self.Input, ones)
  self.Output.Mul(inputAndBias, self.Weight)
  self.Gradient.Apply(
      func (r, c int, v float64) float64 { return self.DActivationFunction(v) },
     self.Output)
  self.Output.Apply(
      func (r, c int, v float64) float64 { return self.ActivationFunction(v) },
      self.Output)
}

// AUGH THE FUCK UP IS HERE
func (self* Layer) Backward(next *Layer) {
  gradient := &mat64.Dense{}
  rows, cols := self.Weight.Dims()
  // Don't look at bias weights from next layer when backpropagating.
  gradient.Mul(self.Weight.View(0, 0, rows - 1, cols), next.Gradient)
  self.Gradient.TCopy(self.Gradient)
  self.Gradient.MulElem(gradient, self.Gradient)
}

func (self* Layer) BackwardOutput(values *mat64.Dense) {
  values.Sub(self.Output, values)
  self.Gradient.MulElem(values, self.Gradient)
  self.Gradient.TCopy(self.Gradient)
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
  deltas := &mat64.Dense{}
  deltas.Mul(self.Gradient, self.Input)
  deltas.TCopy(deltas)
  decay := &mat64.Dense{}
  rows, cols := self.Weight.Dims()
  weight := self.Weight.View(0, 0, rows - 1, cols).(*mat64.Dense)
  if *learningConfiguration.Decay > 0 {
    decay.Scale(*learningConfiguration.Decay, weight)
    deltas.Sub(deltas, decay)
  }
  deltas.Scale(*learningConfiguration.Rate, deltas)
  weight.Sub(weight, deltas)
}

func (self* Layer) DebugString() string {
  return fmt.Sprintf(
      "input: %v\nweight: %v\nname: %v\noutput: %v\ngradient: %v\n",
      mat64.Formatted(self.Input, mat64.Prefix("       ")),
      mat64.Formatted(self.Weight, mat64.Prefix("        ")), self.Name,
      mat64.Formatted(self.Output, mat64.Prefix("        ")),
      mat64.Formatted(self.Gradient, mat64.Prefix("          ")))
}
