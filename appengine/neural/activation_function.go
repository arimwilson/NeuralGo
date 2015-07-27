package neural

import ("math")

func Activate(name ActivationFunction, x float64) float64 {
  if (name == ActivationFunction_LINEAR) {
    return x
  } else if (name == ActivationFunction_RELU) {
    return math.Max(0, x)
  } else if (name == ActivationFunction_LOGISTIC) {
    return 1 / (1 + math.Exp(-x))
  } else if (name == ActivationFunction_TANH) {
    return math.Tanh(x)
  } else {
    return 0
  }
}

func DActivate(name ActivationFunction, y float64) float64 {
  if (name == ActivationFunction_LINEAR) {
    return 1
  } else if (name == ActivationFunction_RELU) {
    if y <= 0 {
      return 0
    }
    return 1
  } else if (name == ActivationFunction_LOGISTIC) {
    logistic := Activate(name, y)
    return logistic * (1 - logistic)
  } else if (name == ActivationFunction_TANH) {
    return 1 - Activate(name, y) * Activate(name, y)
  } else {
    return 0
  }
}

