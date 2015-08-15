package neural

import (
  "github.com/gonum/matrix/mat64";
  "math/rand"
)

type Datapoint struct {
  Features []float64
  Values []float64
}

func min(a, b int) int {
  if a < b {
    return a
  }
  return b
}

func Train(neuralNetwork *Network, datapoints []Datapoint,
           learningConfiguration LearningConfiguration) {
  // Train on some number of iterations of permuted versions of the input.
  for i := 0; i < int(*learningConfiguration.Epochs); i++ {
    perm := rand.Perm(len(datapoints))
    batchSize := int(*learningConfiguration.BatchSize)
    // Batch size 0 means do full batch learning.
    if batchSize == 0 {
      batchSize = len(datapoints)
    }
    for j := 0; j < len(perm); j += batchSize {
      size := min(batchSize, len(perm) - j)
      features := mat64.NewDense(size, len(datapoints[0].Features), nil)
      values := mat64.NewDense(size, len(datapoints[0].Values), nil)
      for k := 0; k < size; k++ {
        features.SetRow(k, datapoints[perm[j + k]].Features)
        values.SetRow(k, datapoints[perm[j + k]].Values)
      }
      neuralNetwork.Forward(features)
      neuralNetwork.Backward(values)
      neuralNetwork.Update(learningConfiguration)
    }
  }
}

func Evaluate(neuralNetwork Network, datapoints []Datapoint) float64 {
  square_error := 0.0
  for _, datapoint := range datapoints {
    output := neuralNetwork.Evaluate(datapoint.Features)
    for i, value := range datapoint.Values {
      square_error += (value - output[i]) * (value - output[i]) / 2
    }
  }
  return square_error / float64(len(datapoints))
}
