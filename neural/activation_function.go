package neural

import (
  "github.com/gonum/matrix/mat64";
  "math"
)

type ActivationFunction func(x mat64.Matrix, y *mat64.Dense)

func NewActivationFunction(name ActivationName) ActivationFunction {
  switch name {
  case ActivationName_LINEAR:
    return func(x mat64.Matrix, y *mat64.Dense) { y.Clone(x) }
  case ActivationName_LOGISTIC:
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 {
        return 1 / (1 + math.Exp(-v))
      }, x)
    }
  case ActivationName_RELU:
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 { return math.Max(0, v) }, x)
    }
  case ActivationName_TANH:
    return func(x mat64.Matrix, y *mat64.Dense) {
      y.Apply(func(r, c int, v float64) float64 { return math.Tanh(v) }, x)
    }
  case ActivationName_SOFTMAX:
    return func(x mat64.Matrix, y *mat64.Dense) {
      r, c := x.Dims()
      for i := 0; i < r; i++ {
        exp_sum := 0.0
        for j := 0; j < c; j++ {
          exp_sum = exp_sum + math.Exp(x.At(i, j))
        }
        for j := 0; j < c; j++ {
          y.Set(i, j, math.Exp(x.At(i, j)) / exp_sum)
        }
      }
    }
  }
  return nil
}

type DActivationFunction func(y mat64.Matrix, x *mat64.Dense)

func NewDActivationFunction(name ActivationName) DActivationFunction {
  switch name {
  case ActivationName_LINEAR:
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 { return 1 }, y)
    }
  case ActivationName_RELU:
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        if v <= 0 {
         return 0
        }
        return 1
      }, y)
    }
  case ActivationName_LOGISTIC:
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        logistic := 1 / (1 + math.Exp(-v))
        return logistic * (1 - logistic)
      }, y)
    }
  case ActivationName_TANH:
    return func(y mat64.Matrix, x *mat64.Dense) {
      x.Apply(func(r, c int, v float64) float64 {
        tanh := math.Tanh(v)
        return 1 - tanh * tanh
      }, y)
    }
  case ActivationName_SOFTMAX:
    return func(y mat64.Matrix, x *mat64.Dense) {
      // TODO(ariw): Finish this.
      r, c := y.Dims()
      for i := 0; i < r; i++ {
        exp_sum := 0.0
        for j := 0; j < c; j++ {
          exp_sum = exp_sum + math.Exp(y.At(i, j))
        }
        for j := 0; j < c; j++ {
          x.Set(i, j, math.Exp(y.At(i, j)) / exp_sum)
        }
      }
    }
  }
  return nil
}

