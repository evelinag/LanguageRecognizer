// The 'FsLab.fsx' script loads the XPlot charting library and other 
// dependencies. It is a smaller & simpler version of the FsLab 
// package that you can get from www.fslab.org.
#load "FsLab.fsx"
open FsLab
open System
open System.IO
open XPlot.GoogleCharts
open DiffSharp.Numerical

// ----------------------------------------------------------------------------
// PART 2. Here, we build a simple neural network (with just a single neuron)
// to distinguish between languages. This has better results than the
// solution in Part 1, but it is a bit more work!
// ----------------------------------------------------------------------------

// STEP #1: First, we need a few things form the first step, most
// importantly the 'getFeatureVector' function that you wrote

let cleanDir = __SOURCE_DIRECTORY__ + "/clean/"
let featuresFile = __SOURCE_DIRECTORY__ + "/features.txt"
let features = File.ReadAllLines(featuresFile)

let getFeatureVector text = __

// ----------------------------------------------------------------------------
// DEMO: To implement the training for our neuron, we'll use a library
// DiffSharp that lets us automatically differentiate functions. Let's
// look at a few examples of how this works! 

// Define a sample function & draw a chart:

let sinSqrt x = sin (3.0 * sqrt x)
Chart.Line [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrt x ]

// We can use 'diff' to differentiate any function!
// Let's do this on 'sinSqrt' and plot the charts

let sinSqrtDiff = diff sinSqrt

Chart.Line
  [ [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrt x ]
    [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrtDiff x ] ]
|> Chart.WithLabels ["function"; "derivative"]

// We can see that the derivative is equal to zero in local
// maxima and local minima of the function

// We can also differentiate functions of multiple parameters,
// but they need to take parameters as arrays of floats

// Example: plot the 'mexican hat' function
let mexicanHat (arr:float[]) = 
  let x, y = arr.[0], arr.[1]
  let r = sqrt(x*x + y*y) + 1e-10
  (sin r) / r;

Chart.Line [ for x in -10.0 .. 0.01 .. 10.0 -> x, mexicanHat [|x;x|] ]

// Mexican hat is a two-dimensial function. To get multi-variable
// differentiation, we use the 'grad' function. Then we can see 
// the direction (how much is the function going up/down) at
// various points in both of the two directions.

// Computing gradient of the 'mexican hat' function:
let mexicanHatDiff = grad mexicanHat

// Evaluating the gradient at specific locations
mexicanHatDiff [| 0.0; 0.0 |]
mexicanHatDiff [| 1.5; 1.5 |]
mexicanHatDiff [| 3.2; 3.2 |]
mexicanHatDiff [| 0.0; 6.0 |]


// ----------------------------------------------------------------------------
// STEP #2: Loading data per pages.
// This time, we load data for two languages and we split them into individual
// pages (training samples). We then build training data where we have the
// expected result (1.0 for one language, 0.0 for the other language) together
// with the feature vectors.

// Split data into individual pages for each of the languages
let allTrainingData = 
    Directory.GetFiles(cleanDir, "*.txt")
    |> Array.map (fun file ->
        let lang = Path.GetFileNameWithoutExtension(file)
        let text = File.ReadAllText(file)
        let sentenceFeatures = 
            text.Split [|'\n'|]
            |> Array.map getFeatureVector
        lang, sentenceFeatures)
    |> dict

// For training the neural network, we need to add 1.0 to the feature vector
// to account for the bias
let prependOne arr = Array.append [| 1.0 |] arr

// Get feature vector for two languages that we want to recognize
// For the first language, the true label is 1
// For the second language, the true label is 0
let lang1 = allTrainingData.["English"] |> Array.map (fun v -> 1.0, prependOne v)
let lang2 = allTrainingData.["French"] |> Array.map (fun v -> 0.0, prependOne v)

// Build the training data set by appending the two arrays
let trainingData = Array.append lang1 lang2


// ----------------------------------------------------------------------------
// STEP #3: Predicting using a neural net & training it.
// The neural network is determined by weights that specify which of the
// features matter for recognizing the two languages. We'll write a prediction
// function (given input & weitghts, predicts the language) and 
// an error function (calculates how good our result is). Then we'll
// write some code to train the neuron (to get the right weights)

// The sigmoid function is used to transform the output of a neuron
// into the [0,1] interval.
// Write the sigmoid function, use the equation given on wikipedia:
// https://en.wikipedia.org/wiki/Sigmoid_function
let sigmoid arg = __

// The prediction function takes all inputs, multiplies it by weights
// and then applies the sigmoid function to the sum of the result:
//
//             f1      f2      f3
//               \     | w2  /
//              w1 \   |   / w3
//                   \ | /
//   sigmoid (f1*w1 + f2*w2 + f3*w3)
//

let predict weights features = __


// The error function takes the training data & current weights and calculates
// how good our result is. That is, for all the items in the input training data, 
// we have the expected label and features.
// For each language features in 'trainingData', we call 'predict weights features' 
// to calculate the predicted label. Then we calculate the Euclidean 
// distance between the predicted label and the true label.
// We sum this over all the training data.
let error trainingData weights = 
   __

// The initial weights for the neuron are generated randomly (this gives 
// bad results :-), but we need something to start the training!)
let initialWeights = 
    let rnd = System.Random()
    Array.init (features.Length + 1) (fun _ -> rnd.NextDouble())

// What is the current error using the random weights?
error trainingData initialWeights

// ----------------------------------------------------------------------------
// STEP #4: Training the neural network.
// To train the neural network, we need to improve the weights. To do this
// we'll calculate the "gradient" of the 'error' function with respect
// to the weights. Then we adapt the weights by "jumping"
// in the right direction by a constant called "eta".

let eta = 0.2

// Compute derivative of the 'error' function, using 'grad'. 
// The 'errorGradient' should be a function taking weights as its
// input (you can use partial function application!)
let errorGradient = __

// Compute the initial error gradient
let gradient = errorGradient initialWeights

// Now we want to minimize the error function which means we'll be going
// against the gradient. 
// We have the gradient for the initial value of weights,
// and we want to calculate new weights such that:
//   newWeight[i] = oldWeight[i] - gradient[i]*eta
// We can nicely do this using the 'Array.map2' function!
let newWeights = __

// Assuming your training improved the weights, 
// the error should be getting smaleler!
error trainingData initialWeights
error trainingData newWeights


// Now we just need to run the weight adaptation that you wrote in loop.
// The following is a recursive function that counts how many times it runs
// and it adapts the weights 5000 times (you can try changning this too).
// You just need to fill in the part that calculates 'newWeights'
let rec gradientDescent steps weights =
  if steps > 5000 then 
    weights
  else
    let gradient = __   // TODO: Calculate new gradient!
    let newWeights = __ // TODO: Calculate new weigths!
    gradientDescent (steps+1) newWeights

// After the training, the error should be much smaller
let trained = gradientDescent 0 initialWeights
error trainingData trained

// Classify two samples from similar languages to see how it works!
let en = prependOne (getFeatureVector "Do you enjoy learning by getting your hands dirty and getting stuck into new concepts and ideas?")
let fr = prependOne (getFeatureVector "De mystérieux bateaux fantômes avec des cadavres à bord découverts sur les côtes du Japon.")

// Predicted value > 0.5 predicts language 1, otherwise language 2
// The closer the value is to 0.5, the less certain the model is
predict trained en
predict trained fr

// ----------------------------------------------------------------------------
// OPEN QUESTIONS

// 1. Change the 'gradientDescent' function to also return the error value
//    in each iteration.


// 2. Plot the error after training to see how long it takes until the value
//    converges. 
//    - How does this depend on the step size 'eta'?


// 3. Observe how the error during training changes when we use the neuron
//    to distinguish the two most different languages (identified in Part 1).
//    How does the error look when we train the neuron to distinguish the two
//    most similar languages? 
//    What would you do to get better performance?
