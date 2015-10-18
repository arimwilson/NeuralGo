package neural

import (
  "github.com/gonum/matrix/mat64";
  "math"
)

type ActivationFunction func(x mat64.Matrix, y *mat64.Dense)

func NewActivationFunction(name ActivationName) ActivationFunction {
  if (name == ActivationName_LINEAR) {
    return func(x mat64.Matrix, y *mat64.Dense) { y.Clone(x) }
  } else if (name == ActivationName_RELU) {
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 { return math.Max(0, v) }, x)
    }
  } else if (name == ActivationName_LOGISTIC) {
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 {
        return 1 / (1 + math.Exp(-v))
      }, x)
    }
  } else {  // TANH
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 { return math.Tanh(v) }, x)
    }
  }
}

type DActivationFunction func(y mat64.Matrix, x *mat64.Dense)

func NewDActivationFunction(name ActivationName) DActivationFunction {
  if (name == ActivationName_LINEAR) {
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 { return 1 }, y)
    }
  } else if (name == ActivationName_RELU) {
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        if v <= 0 {
         return 0
        }
        return 1
      }, y)
    }
  } else if (name == ActivationName_LOGISTIC) {
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        logistic := 1 / (1 + math.Exp(-v))
        return logistic * (1 - logistic)
      }, y)
    }
  } else {  // TANH
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        tanh := math.Tanh(v)
        return 1 - tanh * tanh
      }, y)
    }
  }
}

