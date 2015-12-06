open System
open System.IO
open System.Net
open System.Threading
open System.Text.RegularExpressions

// Override the default .NET limit for 2 connections
System.Net.ServicePointManager.DefaultConnectionLimit <- 20

// Regular expressions for extracting HTML elements :-)
let regexLink = Regex("\<a href=\"/wiki/[^\"]*\"")
let regexTitle = Regex("\<title\>[^\<]*\<")

// ------------------------------------------------------------------
// Extrcting text from Wikipedia
// ------------------------------------------------------------------

/// Extract text from wikipedia page - this skips everything before 
/// and after the first heading, removes all HTML entities between
/// and returns lowercase words in the text
let extractText (html:string) = 
  let sections = html.Split([|"<h1>"; "<h2>"|], StringSplitOptions.None)
  let body = String.concat "" sections.[1 .. sections.Length - 2]
  let words = Regex.Replace(body, "\s*\<[^\>]*>\s*", " ").ToLower().Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
  words

/// Extracts the title of a page and drops the " - Wikipedia"
/// part (which is the same for all pages)
let extractTitle (html:string) = 
  let title = regexTitle.Match(html).Value
  if not (title.Contains("-")) then title.Substring(7, title.Length-8)
  else title.Substring(7, title.LastIndexOf('-')-8)

/// Download Wikipedia page, extract title, words and all links
let download lang (url:string) = async {
  let wc = new WebClient()
  wc.Encoding <- System.Text.UTF8Encoding.UTF8
  let! html = wc.AsyncDownloadString(Uri(url))

  // Extract all links to proper Wikipedia pages
  let allLinks = 
    [ for link in regexLink.Matches(html) do
        let atag = link.Captures.[0].Value
        if not (atag.Contains(":")) then
          let prefix = "http://" + lang + ".wikipedia.org"
          yield prefix + atag.Split('"').[1] ]
  
  // Return the page title together with all the links
  let title = extractTitle html
  let words = extractText html
  return title, words, allLinks }

/// Do a random walk through Wikipedia and download some random
/// articles (making sure that we stay in the same language)
let rec randomWalk lang (rnd:Random) count url = async {
  if count = 0 then return []
  else     
    try
      let! title, words, links = download lang url
      let next = links.[rnd.Next(links.Length)]
      let! rest = randomWalk lang rnd (count-1) next
      return (title, words) :: rest 
    with e -> return [] }
 
/// Create a random number generator (with the specified seed)
/// and use it to crawl Wikipedia from the given URL
let startRandomWalk seed count lang url =
  let rnd = new Random(seed)
  randomWalk lang rnd count url

startRandomWalk 2 2 "en" "http://en.wikipedia.org"
|> Async.RunSynchronously

// ------------------------------------------------------------------
// Get list of Wikipedia languages using HTML type provider,
// download some pages for all languages & save them locally
// ------------------------------------------------------------------

#r "packages/FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type WikiList = HtmlProvider<"http://meta.wikimedia.org/wiki/List_of_Wikipedias">

// All wikis with over 100k articles (with their address & language)
let wikis = 
  [ for wiki in WikiList.GetSample().Tables.``1 000 000+ articles``.Rows do
      yield wiki.Wiki, wiki.Language 
    for wiki in WikiList.GetSample().Tables.``100 000+ articles``.Rows do
      yield wiki.Wiki, wiki.Language ]

// Save the downloaded data into the 'raw' folder.
open System.IO

let (@@) a b = System.IO.Path.Combine(a, b)
let rawRoot = __SOURCE_DIRECTORY__ @@ "raw"
Directory.CreateDirectory(rawRoot)

// Run the download for all wiki pages in parallel and save two files for
// each language - 'lang_info.txt' with page titles and 'lang_words.txt'
// containing the extracted text from the Wikipedia pages
[ for wiki, language in wikis -> async {
    printfn "Starting %s" language 
    let url = "http://" + wiki + ".wikipedia.org"
    let! pages = startRandomWalk 0 50 wiki url 
    let info = 
      [ yield language 
        yield System.String(language |> Seq.map (fun _ -> '=') |> Array.ofSeq)
        yield ""
        yield "Pages:"
        for title, words in pages do
          yield sprintf " - %s (%d words)" title (Seq.length words) ]
    let words = 
        pages 
        |> Seq.map (fun (_, text) -> String.concat " " text) 
        |> String.concat "\n"
    File.WriteAllLines(rawRoot @@ wiki + "_info.txt", info)
    File.WriteAllText(rawRoot @@ wiki + "_words.txt", words)
    printfn "Downloaded %s" language } ]
|> Async.Parallel
|> Async.Ignore
|> Async.Start

// ------------------------------------------------------------------
// Data cleanup - get only languages using latin alphabet, filter
// non-alphabet characters and remove double spaces.
// ------------------------------------------------------------------

open System.Text
open System.Globalization

// Remove diacritics using fancy Unicode magic
let normalize (text:string) = 
  text.Normalize(NormalizationForm.FormKD).ToCharArray()
  |> Array.choose (fun ch ->
      match CharUnicodeInfo.GetUnicodeCategory(ch) with
      | UnicodeCategory.NonSpacingMark -> None
      | _ -> Some(ch))
  |> Array.map (fun ch -> Char.ToLower(ch) |> string)
  |> String.concat ""    

// Get only languages using latin alphabet
let alphabet = ' '::['a' .. 'z'] |> set
let alphabetNewLine = Set.add '\n' alphabet
let regexMultiSpace = Regex(" +")

let latinData = 
  [ for file in Directory.GetFiles(rawRoot, "*words.txt") do
      // Read the pages saved into the RAW file
      let name = Path.GetFileNameWithoutExtension(file)
      let lang = File.ReadAllLines(file.Replace("words", "info")).[0]
      let text = File.ReadAllText(file) |> normalize
      let newText = String(text.ToCharArray() |> Array.filter (alphabetNewLine.Contains))
      let newText = regexMultiSpace.Replace(newText, " ")

      // Filter out non-latin alphabet languages
      if newText.Length > 0 then 
        let mostCommonLetters = 
            text.ToCharArray() 
            |> Seq.filter (Char.IsLetter)
            |> Seq.countBy id 
            |> Seq.sortBy snd
        if not (Seq.isEmpty mostCommonLetters) && 
           alphabet.Contains (fst (Seq.last mostCommonLetters)) then
          yield lang, newText ]

// Save the data to the 'clean' folder
let cleanRoot = __SOURCE_DIRECTORY__ @@ "clean"
Directory.CreateDirectory(cleanRoot)
for lang, text in latinData do
  File.WriteAllText(cleanRoot @@ (lang + ".txt"), text)

// Create file with all features (two character letters)
// (this generates all possible pairs, even though some of them are not too frequent)
let allFeatures = 
    alphabet 
    |> Seq.collect (fun ch -> alphabet |> Seq.map (fun a -> (ch, a)))
    |> Seq.filter (fun (ch1, ch2) -> ch1 <> ' ' || ch2 <> ' ')  // remove double space
    |> Seq.map (fun (ch1, ch2) -> String [| ch1; ch2 |])
    |> Array.ofSeq

// File.WriteAllLines(__SOURCE_DIRECTORY__ @@ "features.txt", allFeatures)

let allPairs = 
  latinData
  |> List.map snd
  |> String.concat "\n"
  |> Seq.pairwise 
  |> Seq.countBy id
  |> Array.ofSeq
  |> Array.sortBy snd
  |> Array.rev

let topFeatures = 
  allPairs 
  |> Seq.take 100
  |> Seq.map (fun ((ch1, ch2), _) -> String [| ch1; ch2 |])
  |> Array.ofSeq

File.WriteAllLines(__SOURCE_DIRECTORY__ @@ "features.txt", topFeatures)