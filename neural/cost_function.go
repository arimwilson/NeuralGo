// TODO(ariw): Use Cost functions here in learn's Evaluate method instead of
// MSE.

package neural

import (
  "github.com/gonum/matrix/mat64";
  // "math"
)

type ErrorFunction interface {
  Cost(values mat64.Matrix, outputs mat64.Matrix) float64
  Deltas(values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix
}

type QuadraticErrorFunction struct {
}
func (m* QuadraticErrorFunction) Cost(
    values mat64.Matrix, outputs mat64.Matrix) float64 {
  return 0
}
func (m* QuadraticErrorFunction) Deltas(
    values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix {
  return outputs
}

type CrossEntropyErrorFunction struct {
}
func (m* CrossEntropyErrorFunction) Cost(
    values mat64.Matrix, outputs mat64.Matrix) float64 {
  return 0
}
func (m* CrossEntropyErrorFunction) Deltas(
    values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix {
  return outputs
}

func NewErrorFunction(name ErrorName) ErrorFunction {
  switch name {
  case ErrorName_QUADRATIC:
    return new(QuadraticErrorFunction)
  case ErrorName_CROSS_ENTROPY:
    return new(CrossEntropyErrorFunction)
  }
  return nil
}

