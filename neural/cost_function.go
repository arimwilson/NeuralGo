package neural

import (
  "github.com/gonum/matrix/mat64";
  "math"
)

type ErrorFunction interface {
  Cost(values mat64.Matrix, outputs mat64.Matrix) float64
  Deltas(values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix
}

struct MeanSquaredErrorFunction {
  Cost(values mat64.Matrix, outputs mat64.Matrix) float64 {
  }

  Deltas(values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix {
  }
}

struct CrossEntropyErrorFunction {
  Cost(values mat64.Matrix, outputs mat64.Matrix) float64 {
  }

  Deltas(values mat64.Matrix, outputs mat64.Matrix) mat64.Matrix {
  }
}

func NewErrorFunction(name ErrorName) ErrorFunction {
  if (name == ErrorName_MEAN_SQUARED) {
    return MeanSquaredErrorFunction()
  } else {
    return CrossEntropyErrorFunction()
  }
}

