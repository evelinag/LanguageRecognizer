// The 'FsLab.fsx' script loads the XPlot charting library and other 
// dependencies. It is a smaller & simpler version of the FsLab 
// package that you can get from www.fslab.org.
#load "FsLab.fsx"
open FsLab
open System
open System.IO
open XPlot.GoogleCharts

// ----------------------------------------------------------------------------
// PART 1. We use frequency of character pairs to classify languages. E.g. in 
// English text "the three " there is 2x "th", 2x "e " and 1x other pairs.
// ----------------------------------------------------------------------------

// The 'featuresFile' file contains all letter pairs that we want to
// use for language detection (we do not use all pairs of all 
// letters, but only smaller number of more frequent ones). The
// 'cleanDir' contains cleaned Wikipedia pages in various languages.

let featuresFile = __SOURCE_DIRECTORY__ + "/features.txt"
let cleanDir = __SOURCE_DIRECTORY__ + "/clean/"


// The first task is to calculate probabilities of letter pairs in a text
// We only want to use letter pairs specified in 'featuresFile' and we
// want to do this for data from "English.txt" (containing Wiki pages)

let sampleFile = cleanDir + "English.txt"


// ----------------------------------------------------------------------------
// STEP #1: Read the input files. We need to read individual *lines*
// from the feature file and *all text* from the sampleFile. 

// Use "File.ReadAllLines" to get the features
let features = File.ReadAllLines featuresFile

// Use "File.ReadAllText" to get all the sample text
let sampleText = File.ReadAllText sampleFile


// ----------------------------------------------------------------------------
// STEP #2: Take all letter pairs from the sample text and calculate
// how many times they appear in the input. To do this, there is a
// number of functions that will help you:

// [======= DEMO =======]
module DEMO1 =

  // Seq.pairwise returns a sequence of pairs from the input
  Seq.pairwise "hello"  

  // We can write the same thing using the "|>" operator. This
  // is nice if we want to perform more operations in a pipeline
  "hello" |> Seq.pairwise

  // For example, we can take pairs and then convert the two
  // characters back to a two-letter string. Here, the "pairwise"
  // function creates a tuple and "(c1, c2)" decomposes it into
  // the first and the second element
  "hello"
  |> Seq.pairwise
  |> Seq.map (fun (c1, c2) -> string c1 + string c2)

  // Count how many times each letter appears in a word. This
  // returns a sequence of pairs "seq<char * int>" containing
  // the key (the letter itself) and the count
  "hello"
  |> Seq.countBy (fun c -> c)
// [======= /DEMO =======]


let counts = 
  // Now you have all you need to calculate counts of pairs:
  //  - Take the 'sampleText'
  //  - Use 'Seq.pairwise' to turn it into pairs of letters
  //  - Use 'Seq.map' to turn the letter pairs into strings
  //  - Use 'Seq.countBy' to count how many times each pair appears
  sampleText
  |> Seq.pairwise
  |> Seq.map (fun (c1,c2) -> string c1 + string c2)
  |> Seq.countBy id


// ----------------------------------------------------------------------------
// DEMO: Draws a column chart showing the 50 most common letter pairs

counts 
|> Array.ofSeq 
|> Array.sortBy (fun (_, count) -> -count) 
|> Seq.take 50 
|> Chart.Column

// ----------------------------------------------------------------------------
// STEP #3: Calculating probability vector 
// So far, we have total counts of letter pairs. This is not enough, because
// it depends on the total length of the text. So, we need to get probabilities
// instead and we also only care about letter pairs in the 'features' file taht
// we loaded earlier. To do this, we want to iterate over the 'features' and
// for each feature, calculate the number of occurrences divided by the total
// number of letter pairs in the input. If the feature does not appear at all,
// we return a small number 1e-10 instead (to avoid division by zero later)

// [======= DEMO =======]
module DEMO2 =
  // To efficiently find counts of letter pairs, we'll use a lookup table.
  // You can create one from a sequence of pairs using the F# 'dict' function
  let words = [ ("hi", 10); ("ola", 20) ]
  let wordsLookup = dict words
  
  // Now you can check if the lookup table contans key 
  // and get the value for a given key using lookup
  wordsLookup.ContainsKey("ahoj")
  wordsLookup.["ola"]

  // Note that F# 'if' is an expression that returns the value of the
  // true or fals branch (more like "cond ? true : false" in C#/C++)
  let count = if wordsLookup.ContainsKey("ahoj") then 0 else 1

  // Aside from this, you'll need the 'Array.map' function. This is like 'Seq.map'
  // but it returns an array (which is more efficient for what we do here). This 
  // returns array with the lengths of the features (they should all be 2 letters)
  features |> Array.map (fun f -> String.length f)
// [======= /DEMO =======]


// How many pairs are there in total? The length of the text - 1. This 
// is an integer, but we need floating-point so that we can calculate with it.
let total = float (String.length sampleText - 1)

// Create a lookup table from 'counts' (call it 'countLookup'). Then
// calculate probability for all features using "features |> Array.map" and
// returning 1e-10 if the feature is not found or "count / total" otherwise

let countLookup = dict counts

let probabilities = 
    features
    |> Array.map (fun s ->
        let f = countLookup.[s]
        (float f)/total)


// ----------------------------------------------------------------------------
// DEMO: Draws a column chart showing the probabilities of features
// (Make this very wide so that we can see anyting)

Chart.Column(Seq.zip features probabilities)
|> Chart.WithSize (1200, 500)


// ----------------------------------------------------------------------------
// STEP #4: Making a reusable F# function
// Now we implemented all the logic we need, but we want to call this on
// all the languages, so we need to wrap this into a function. To do this,
// copy the code into the body of a function (and indent everything further)
// Also, change it so that it processes the 'text' given as a parameter

let getFeatureVector text = 
    let counts = 
      text
      |> Seq.pairwise
      |> Seq.map (fun (c1,c2) -> string c1 + string c2)
      |> Seq.countBy id

    let total = float (String.length text - 1)

    let countLookup = dict counts

    features
    |> Array.map (fun s ->
        if countLookup.ContainsKey(s) then 
            let f = countLookup.[s]
            (float f)/total
        else 1e-15)




// Now, let's run your 'getFeatureVector' function on all languages in the
// cleaned-up folder to build a lookup that gives us "feature vector"
// for every language. This takes some time. You can enable timing using
// #time and you can parallelize this by replacing "Array.map" with
// "Array.Parallel.map" - a handy parallel version from the F# library!

#time

let languageFeatures = 
  Directory.GetFiles(cleanDir, "*.txt")
  |> Array.map (fun file ->
      Path.GetFileNameWithoutExtension(file),
      getFeatureVector (File.ReadAllText(file)) )

let byLanguage = dict languageFeatures

// DEMO: Draw a column chart comparing three languages in a single chart
// (We should see that Portuguese is closer to Spanish than English)

[ Seq.zip features byLanguage.["English"]
  Seq.zip features byLanguage.["German"]
  Seq.zip features byLanguage.["French"] ]
|> Chart.Column
|> Chart.WithLabels ["English"; "German"; "French"]
|> Chart.WithSize (2000, 500)

// ----------------------------------------------------------------------------
// STEP #5: Calculating the distance between languages
// To calculate "distance" between two feature vectors, we'll take the 
// Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance).
//
// Given two feature vectors [a1; ..; an] and [b1; ..; bn], 
// we want to take: (a1-b1)**2.0 + .. + (an-bn)**2.0.
// (the ** operator is the F# power operator, used here to get a square)

// [======= DEMO =======]
module DEMO3 =
  // Calculating square using the ** operator:
  let square = 4.0 ** 2.0

  // A very useful function we can use here is 'Array.map2'. This takes
  // two arrays nd it calls the specified function for the matching
  // elements of the array:
  let arr1 = [| 1.0; 2.0; 3.0 |]
  let arr2 = [| 3.0; 2.0; 1.0 |]
  Array.map2 (fun a b -> a + b) arr1 arr2

  // You'll also need 'Array.sum' to get the total distance
  Array.sum arr1
// [======= /DEMO =======]


let distance (features1:float[]) (features2:float[]) = 
    // Calculate the Euclidean distance using Array.map2 and Array.sum
    Array.map2 (fun a b -> (a-b) ** 2.0) features1 features2
    |> Array.sum


// Check how our distance function works for a few languages
distance (byLanguage.["English"]) (byLanguage.["Spanish"])
distance (byLanguage.["French"]) (byLanguage.["English"])
distance (byLanguage.["English"]) (byLanguage.["Czech"])


// Given some text, we can now classify it by finding the language with the
// most similar feature vector. To do this, we need to call 
// 'getFeatureVector' on the given 'text', sort the 'langaugeFeatures' by
// the distance between their feature vector and the one we got from the 
// text and then pick the first one. You'll need "Array.sortBy" and
// "Seq.head" functions to do this.

let classifyLanguage text =
    let textFeatures = getFeatureVector text
    languageFeatures
    |> Seq.sortBy (fun (lang, features) -> distance textFeatures features)
    |> Seq.head
    |> fst

// Some examples    
classifyLanguage "tohle je nejaky text napsany v ceskem jazyce"
classifyLanguage "the recent expose of amazons aggressive corporate culture was a cause of consternation to many but plenty of others couldnt see what the problem was"
classifyLanguage "Negotiators paving the way for a global climate change agreement in Paris have cleared a major hurdle, producing a draft accord in record time and raising hopes that a full week of minister-led talks can now clinch a deal despite many sticking points."
classifyLanguage "Unbequeme Politiker werden ermordet, Wahlkreise radikal umgestellt, wie es der sozialistischen Regierung passt: Trotzdem steht in Venezuela die Opposition vor dem Wahlsieg."
classifyLanguage "us stock markets follow global plunge as china concerns deepen"


// ----------------------------------------------------------------------------
// OPEN QUESTIONS

// Which languages are the closest? Which ones are the most different?
// 1. For each language, find the closest other language. 
//    Use 'languageFeatures' and 'distance' to find the closest language for each 
//    language.

let closestLanguages = 
    languageFeatures
    |> Array.map (fun (lang, features) ->
        let closest =
            languageFeatures
            |> Array.map (fun (lang2, features2) -> 
                if lang = lang2 then (lang2, Double.PositiveInfinity)
                else (lang2, distance features features2))
            |> Array.minBy snd
        lang, fst closest, snd closest)

// 2. Which languages are the closest in terms of their absolute distance?



// 3. Which languages are the most different?
//    And which language is the most different from all other languages?
let differentLanguages = 
    languageFeatures
    |> Array.map (fun (lang, features) ->
        let closest =
            languageFeatures
            |> Array.map (fun (lang2, features2) -> 
                if lang = lang2 then (lang2, - infinity)
                else (lang2, distance features features2))
            |> Array.maxBy snd
        lang, fst closest, snd closest)
differentLanguages |> Array.maxBy (fun (_,_,d) -> d)

// 4. Which feature distinguishes English the most from all other languages?
let differences = 
    languageFeatures
    |> Array.map (fun (lang, features) -> 
        Array.map2 (fun l1 l2 -> (l1 - l2)**2.0) byLanguage.["English"]  features)

Array.init differences.[0].Length (fun i ->
    differences |> Array.map (fun ds -> ds.[i])
    |> Array.sum)
|> Array.zip features
|> Array.sortBy (fun (_,d) -> -d)


// ----------------------------------------------------------------------------
// Bonus: visualize the languages using Principal Component Analysis (PCA)
// Read up on PCA here: http://setosa.io/ev/principal-component-analysis/
// How to compute PCA manually: follow the explanation on wikipedia
//   https://en.wikipedia.org/wiki/Principal_component_analysis#Computing_PCA_using_the_covariance_method
// The following code follows the wikipedia computation, using Math.NET library:

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

let dataMatrix = 
    languageFeatures
    |> Array.map snd
    |> DenseMatrix.ofColumnArrays

let n = dataMatrix.ColumnCount |> float

// Compute empirical mean
let center = 
    dataMatrix.RowSums() / n

// Normalize data to zero mean
let centeredMatrix = 
    let sumMatrix = 
        Array.init dataMatrix.ColumnCount (fun _ -> center)
        |> DenseMatrix.OfColumnVectors
    dataMatrix - sumMatrix

// Compute covariance matrix
let covarianceMatrix = 
    1.0/(n-1.0) * (centeredMatrix * centeredMatrix.Transpose())

// Eigenvalue decomposition
let evd = covarianceMatrix.Evd()

// Extract d eigenvectors
let eigenvectors d = 
    let m = evd.EigenVectors
    m.[0..m.RowCount-1, m.ColumnCount-d..m.ColumnCount-1]

// Project the data to the d eigendimensions
let projection d = (eigenvectors d).Transpose() * centeredMatrix

// Visualize 2D projected data
let projectedData = (projection 2).ToRowArrays()
let dataToPlot = Array.zip projectedData.[0] projectedData.[1]
let labels = Array.map fst languageFeatures

Chart.Scatter [for p in dataToPlot->[p]]
|> Chart.WithLabels(labels)
|> Chart.WithXTitle("Principal component 1")
|> Chart.WithYTitle("Principal component 2")

// Note that in larger applications, it is more numerically stable to compute
// PCA using the singular value decomposition
// see https://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition
