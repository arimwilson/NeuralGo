package neural

import ("math/rand")

type Datapoint struct {
  Features []float64
  Values[] float64
}

func Train(neuralNetwork *Network, datapoints []Datapoint,
           learningConfiguration LearningConfiguration) {
  // Train on some number of iterations of permuted versions of the input.
  for i := 0; i < int(*learningConfiguration.Epochs); i++ {
    perm := rand.Perm(len(datapoints))
    for _, index := range perm {
      neuralNetwork.Forward(datapoints[index].Features)
      neuralNetwork.Backward(datapoints[index].Values)
      if (index + 1) % int(*learningConfiguration.BatchSize) == 0 {
        neuralNetwork.Update(learningConfiguration)
      }
    }
    // In case we didn't finish a mini batch...
    neuralNetwork.Update(learningConfiguration)
  }
}

func Evaluate(neuralNetwork Network, datapoints []Datapoint) float64 {
  square_error := 0.0
  for _, datapoint := range datapoints {
    output := neuralNetwork.Forward(datapoint.Features)
    for i, value := range datapoint.Values {
      square_error += (value - output[i]) * (value - output[i]) / 2
    }
  }
  return square_error / float64(len(datapoints))
}
