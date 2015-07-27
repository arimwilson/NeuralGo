// Feed-forward neural network training & execution on a simple supervised
// regression problem.
//
// Sample usage:
// go run cmdline.go -serialized_network_file network.txt -training_file training.txt -testing_file testing.txt

package main

import ("encoding/json"; "flag"; "fmt"; "github.com/golang/protobuf/proto";
        "io/ioutil"; "log"; "math/rand"; "time"; "./appengine/neural")

var serializedNetworkFlag = flag.String(
  "serialized_network", "", "File with JSON-formatted NetworkConfiguration.")
var trainingExamplesFlag = flag.String(
  "training_file", "",
  "File with JSON-formatted array of training examples with values.")
var trainingIterationsFlag = flag.Int(
  "training_iterations", 1000, "Number of training iterations.")
var trainingSpeedFlag = flag.Float64(
  "training_speed", 0.001, "Speed of training.")
var testingExamplesFlag = flag.String(
  "testing_file", "",
  "File with JSON-formatted array of testing examples with values.")

func ReadDatapointsOrDie(filename string) []neural.Datapoint {
  bytes, err := ioutil.ReadFile(filename)
  if err != nil {
    log.Fatal(err)
  }
  datapoints := make([]neural.Datapoint, 0)
  err = json.Unmarshal(bytes, &datapoints)
  if err != nil {
    log.Fatal(err)
  }
  return datapoints
}

func main() {
  flag.Parse()
  rand.Seed(time.Now().UTC().UnixNano())

  // Set up neural network using flags.
  var neuralNetwork *neural.Network
  trainingExamples := ReadDatapointsOrDie(*trainingExamplesFlag)
  byteNetwork, err := ioutil.ReadFile(*serializedNetworkFlag)
  if err != nil {
    log.Fatal(err)
  }
  neuralNetwork = new(neural.Network)
  neuralNetwork.Deserialize(byteNetwork)
  // If synapse weights aren't specified, randomize them.
  if neuralNetwork.Layers[0].Neurons[0].InputSynapses[0].Weight == 0 {
    neuralNetwork.RandomizeSynapses()
  }
  testingExamples := ReadDatapointsOrDie(*testingExamplesFlag)

  // Train the model.
  neural.Train(neuralNetwork, trainingExamples, *trainingIterationsFlag,
               *trainingSpeedFlag)

  // Test & print model:
  fmt.Printf("Training error: %v\nTesting error: %v\nNetwork: %v\n",
             neural.Evaluate(*neuralNetwork, trainingExamples),
             neural.Evaluate(*neuralNetwork, testingExamples),
             string(neuralNetwork.Serialize()))
}
