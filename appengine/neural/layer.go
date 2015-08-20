package neural

import (
  "fmt";
  "github.com/gonum/matrix/mat64"
)

func NewLayer(name ActivationName, inputs int, outputs int,
              weight []float64) *Layer {
  layer := new(Layer)
  layer.Name = name
  layer.ActivationFunction = NewActivationFunction(layer.Name)
  layer.DActivationFunction = NewDActivationFunction(layer.Name)

  layer.Weight = mat64.NewDense(inputs + 1, outputs, weight)
  layer.Output = &mat64.Dense{}
  layer.Deltas = &mat64.Dense{}
  layer.Derivatives = &mat64.Dense{}
  return layer
}

type Layer struct {
  Name ActivationName
  ActivationFunction ActivationFunction
  DActivationFunction DActivationFunction

  Weight *mat64.Dense  // (inputs + 1) x outputs
  Input *mat64.Dense  // examples x (inputs + 1)
  Output *mat64.Dense  // examples x outputs
  Deltas *mat64.Dense  // outputs x examples
  Derivatives *mat64.Dense  // outputs x examples
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
  self.Derivatives = &mat64.Dense{}
  self.Derivatives.Apply(
     func (r, c int, v float64) float64 { return self.DActivationFunction(v) },
     self.Output)
  self.Derivatives.TCopy(self.Derivatives)
  self.Output.Apply(
      func (r, c int, v float64) float64 { return self.ActivationFunction(v) },
      self.Output)
}

func (self* Layer) Backward(next *Layer) {
  rows, cols := next.Weight.Dims()
  // Don't look at bias weights from next layer when backpropagating.
  self.Deltas.Mul(next.Weight.View(0, 0, rows - 1, cols), next.Deltas)
  self.Deltas.MulElem(self.Deltas, self.Derivatives)
}

func (self* Layer) BackwardOutput(values *mat64.Dense) {
  values.Sub(self.Output, values)
  values.TCopy(values)
  self.Deltas.MulElem(values, self.Derivatives)
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
  deltas := &mat64.Dense{}
  deltas.Mul(self.Deltas, self.Input)
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
      "name: %v\nweight: %v\ninput: %v\noutput: %v\ndeltas: %v\nderivatives: " +
      "%v\n", self.Name,
      mat64.Formatted(self.Weight, mat64.Prefix("        ")),
      mat64.Formatted(self.Input, mat64.Prefix("        ")),
      mat64.Formatted(self.Output, mat64.Prefix("        ")),
      mat64.Formatted(self.Deltas, mat64.Prefix("        ")),
      mat64.Formatted(self.Derivatives, mat64.Prefix("             ")))
}
