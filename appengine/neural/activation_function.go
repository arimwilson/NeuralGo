package neural

import (
  "math"
)

type ActivationFunction func(x float64) float64

func NewActivationFunction(name ActivationName) ActivationFunction {
  if (name == ActivationName_LINEAR) {
    return func(x float64) float64 {
      return x
    }
  } else if (name == ActivationName_RELU) {
    return func(x float64) float64 {
      return math.Max(0, x)
    }
  } else if (name == ActivationName_LOGISTIC) {
    return func(x float64) float64 {
      return 1 / (1 + math.Exp(-x))
    }
  } else if (name == ActivationName_TANH) {
    return func(x float64) float64 {
      return math.Tanh(x)
    }
  } else {
    return func(x float64) float64 {
      return 0
    }
  }
}

type DActivationFunction func(y float64) float64

func NewDActivationFunction(name ActivationName) DActivationFunction {
  if (name == ActivationName_LINEAR) {
    return func(y float64) float64 {
      return 1
    }
  } else if (name == ActivationName_RELU) {
    return func(y float64) float64 {
      if y <= 0 {
        return 0
      }
      return 1
    }
  } else if (name == ActivationName_LOGISTIC) {
    return func(y float64) float64 {
      logistic := 1 / (1 + math.Exp(-y))
      return logistic * (1 - logistic)
    }
  } else if (name == ActivationName_TANH) {
    return func(y float64) float64 {
      tanh := math.Tanh(y)
      return 1 - tanh * tanh
    }
  } else {
    return func(y float64) float64 {
      return 0
    }
  }
}

