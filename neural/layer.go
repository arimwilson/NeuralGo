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
  Ones *mat64.Dense  // examples x 1
  Output *mat64.Dense  // examples x outputs
  Deltas *mat64.Dense  // outputs x examples
  Derivatives *mat64.Dense  // outputs x examples
}

// TODO(ariw): Delete any matrix creation in layer operations.
func (self* Layer) Forward(previous *Layer) {
  self.resetForExamples(previous)
  self.Input = previous.Output
  var inputAndBias mat64.Dense
  inputAndBias.Augment(self.Input, self.Ones)  // Add bias to input.
  self.Output.Mul(&inputAndBias, self.Weight)
  self.DActivationFunction(self.Output.T(), self.Derivatives)
  self.ActivationFunction(self.Output, self.Output)
}

func (self* Layer) Backward(next *Layer) {
  rows, cols := next.Weight.Dims()
  // Don't look at bias weights from next layer when backpropagating.
  self.Deltas.Mul(next.Weight.View(0, 0, rows - 1, cols), next.Deltas)
  self.Deltas.MulElem(self.Deltas, self.Derivatives)
}

func (self* Layer) BackwardOutput(values *mat64.Dense) {
  values.Sub(self.Output, values)
  self.Deltas.MulElem(values.T(), self.Derivatives)
}

func (self* Layer) Update(learningConfiguration LearningConfiguration) {
  var deltas mat64.Dense
  deltas.Mul(self.Deltas, self.Input)
  rows, cols := self.Weight.Dims()
  weight := self.Weight.View(0, 0, rows - 1, cols).(*mat64.Dense)
  if *learningConfiguration.Decay > 0 {
    var decay mat64.Dense
    decay.Scale(*learningConfiguration.Decay, weight)
    deltas.Sub(&deltas, decay.T())
  }
  deltas.Scale(*learningConfiguration.Rate, &deltas)
  weight.Sub(weight, deltas.T())
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

// Check if we need to reset internal state for this activation of the network.
func (self* Layer) resetForExamples(previous *Layer) {
  previousExamples, _ := self.Output.Dims()
  examples, _ := previous.Output.Dims()
  if previousExamples != examples {
    self.Output.Reset()
    self.Deltas.Reset()
    self.Derivatives.Reset()
    ones := make([]float64, examples)
    for i, _ := range ones {
      ones[i] = 1.0
    }
    self.Ones = mat64.NewDense(examples, 1, ones)
  }
}
