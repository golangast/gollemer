// gen_intent_data writes the intent-classifier training corpus:
// for each chat action, beginner-style phrasings generated from
// templates with slot filling ({url} {name} {concept} {pkg}
// {target} {path} {route}), multiplied by politeness prefixes,
// question marks, and typo noise. The held-out verdict file
// (data/training/intent_heldout.jsonl) is handwritten separately
// and never generated from these templates.
//
// Run: go run ./scripts/gen_intent_data
package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

type pair struct {
	Input  string `json:"input"`
	Intent string `json:"intent"`
}

var slots = map[string][]string{
	"url":     {"github.com/fgm/izidic", "github.com/golangast/gollemer", "github.com/mautrix/go", "github.com/spf13/cobra", "github.com/gin-gonic/gin", "https://github.com/fgm/izidic", "https://github.com/mautrix/go"},
	"name":    {"Names", "Freeze", "MakeRequest", "New", "Total", "Add", "Get", "Resolve", "Store", "runClassic", "parseFlags", "HandleRequest", "Container", "Client", "Service", "Counter"},
	"name2":   {"Get", "Total", "Freeze", "MustParam", "Store", "Names", "NewClient", "Validate", "Close", "Open", "parseConfig", "handleGet"},
	"concept": {"mutex", "goroutine", "channel", "interface", "error", "map", "slice", "defer", "context", "generics", "pointer", "struct"},
	"pkg":     {"json", "sync", "errors", "fmt", "http", "os", "time", "context", "strings"},
	"target":  {"chat", "eval", "smarter", "train-go", "build", "test", "train-social", "help", "train-gocode"},
	"path":    {"~/projects/foo", "./myproject", "~/Downloads/foo", "/home/a/code/bar", "."},
	"route":   {"/users", "/health", "/items", "/login", "/products"},
}

var templates = map[string][]string{
	"tour": {
		"the lay of the land", "lay of the land", "show me the lay of the land", "get my bearings", "bearings please",
		"i just cloned this", "just cloned this repo", "no idea where to start", "i have no idea where to start", "im lost in this codebase", "i dont know this codebase at all", "orient me", "get me oriented", "orientation please", "new here, show me around",
		"tour", "/tour", "tour {url}", "/tour {url}", "give me a tour", "give me the tour",
		"take the tour", "start the tour", "show me around", "show me around {url}",
		"walk me through the project", "walk me through this project", "walk me through the repo",
		"tour this", "tour it", "tour this project", "guided tour", "i want a tour",
		"can i get a tour", "walk through the whole thing with me",
	},
	"glossary": {
		"types i dont recognize", "types i dont know", "words i dont recognize", "i dont recognize these types", "unfamiliar types",
		"vocabulary", "vocabulary list", "the vocabulary", "word list", "list the words", "words i need to know",
		"glossary", "/glossary", "the glossary", "project glossary", "what are the main types",
		"what are the key types", "define the types", "define the main types", "key terms",
		"what are the key terms", "the dictionary", "project dictionary", "main types",
		"what words do i need to know", "list the important types", "types i should know",
	},
	"learn_path": {
		"learn path", "/learn-path", "learning path", "the learn path", "the learning path",
		"what should i learn first", "what do i learn first", "where should i start learning",
		"learning order", "what order should i learn", "what order should i learn this in",
		"teach me in order", "what do i learn next", "curriculum for this project",
	},
	"example": {
		"show me an example", "give me an example", "an example please", "examples please",
		"who calls {name}", "what calls {name}", "who uses {name}", "callers of {name}",
		"example {name}", "/example {name}", "example of {name}", "show me an example of {name}",
		"show me examples of {name}", "how is {name} used", "where is {name} used",
		"show {name} in use", "usage of {name}", "real usage of {name}",
		"an example for {name}", "examples for {name}",
	},
	"watch": {
		"fire it up", "fire this up", "does it run", "does this run", "does this thing run", "run it and show me", "run it and show me what happens", "start it up",
		"watch", "/watch", "watch it", "watch it run", "run it", "run this", "run the project",
		"run this project", "run the program", "see it run", "let me see it run",
		"try running it", "run the tests", "execute it", "launch it", "start the program",
	},
	"analyze": {
		"analyze", "/analyze", "analyze this project", "analyze {url}", "analyze {path}",
		"look at {url}", "look at this repo {url}", "explain this repo", "explain {url}",
		"read this codebase", "read {url}", "check out {url}", "what is this project",
		"analyze this", "take a look at {url}", "take a look at this project",
	
		"analyze the folder on my desktop", "analyze that project on my desktop", "check the folder on my desktop",},
	"where": {
		"where for {url}", "where would i add a retry helper", "where would i add {concept} support",
		"where do i add a new feature", "where should this go", "where does a new handler go",
		"where would i put a helper", "where to add logging", "where can i add a flag",
		"where should i edit for this issue {url}", "where do i change things for {url}",
	},
	"whatis": {
		"/whatis {name}", "what is {name}", "what does {name} do", "whats {name}",
		"explain {name}", "tell me about {name}", "what is a {name}", "describe {name}",
		"what does the {name} type do", "meaning of {name}",
	},
	"show_func": {
		"show me {name}", "show {name}", "read the source of {name}", "source of {name}",
		"print {name}", "let me see {name}", "display {name}", "code for {name}",
	},
	"walkthrough": {
		"trace what {name} does", "trace through {name}", "what does {name} do step by step",
		"explain how {name} works", "show me how {name} works", "how {name} works", "explain {name} slowly",
		"walk me through {name}", "trace {name}", "walk through {name}",
		"step by step through {name}", "explain {name} step by step", "how does {name} work",
		"take me through {name}",
	},
	"find": {
		"is there code that handles {concept}", "code that handles {concept}", "is there anything for {concept}", "does this have {concept}",
		"/find {concept}", "find {concept}", "find code about {concept}", "search for {concept}",
		"search the code for {concept}", "where is {concept} handled", "look for {concept}",
		"find functions about {concept}", "anything about {concept}",
	},
	"fix": {
		"im getting errors", "im getting cannot use errors", "getting a compile error", "it wont compile", "the compiler says cannot use", "im getting an error when i build",
		"/fix undefined: foo", "fix this error", "fix undefined: {name}", "fix cannot use error",
		"fix imported and not used", "help with this error", "what does this error mean",
		"fix the build error", "why wont this compile", "fix my error",
	},
	"review": {
		"any code smells in {name}", "code smells in {name}", "any problems in {name}", "any issues with {name}", "is {name} written well",
		"/review {name}", "review {name}", "review the {name} function", "check {name}",
		"is {name} ok", "look over {name}", "critique {name}", "code review {name}",
	},
	"comment": {
		"/comment {name}", "comment {name}", "write a comment for {name}",
		"document {name}", "add a doc comment to {name}", "doc comment for {name}",
		"generate a comment for {name}",
	},
	"test_scaffold": {
		"/test {name}", "test {name}", "write a test for {name}", "test skeleton for {name}",
		"scaffold a test for {name}", "generate a test for {name}", "make a test for {name}",
		"unit test for {name}",
	},
	"arch": {
		"how is this project put together", "how is this put together", "how is this organized", "how is this project organized", "whats the organization here",
		"/arch", "arch", "architecture", "the architecture", "summarize the architecture",
		"architecture summary", "how is this structured", "project structure",
		"overall structure", "structure summary",
	},
	"why_import": {
		"/why-import {pkg}", "why import {pkg}", "why does this import {pkg}",
		"what uses {pkg}", "why is {pkg} imported", "where is {pkg} used",
		"what uses the {pkg} package",
	},
	"build_api": {
		"/build-api {route}", "build an api", "build an api with {route}", "build api {route}",
		"compose an api", "make an http api", "create an api with {route} and {route}",
		"build me an api",
	},
	"learn_pieces": {
		"/learn-pieces", "learn pieces", "learn the pieces", "learn pieces from docs",
		"extract pieces", "learn from the docs", "get pieces from documentation",
	},
	"uexplain": {
		"/uexplain {name}", "uexplain {name}", "model explain {name}",
		"use the model on {name}", "model explanation of {name}",
		"what does the model say about {name}",
	},
	"export_understanding": {
		"/export-understanding", "export understanding", "export the understanding pairs",
		"generate understanding pairs", "export structure pairs", "make training pairs from structure",
	},
	"flow": {
		"/flow count with a mutex", "flow count with a mutex", "/flow {concept}",
		"run the flow", "flow this", "one idea end to end", "pipeline for {concept}",
	},
	"help": {
		"list everything you can do", "list what you can do", "everything you can do", "all the things you can do",
		"what are you able to do", "what all can you do", "your abilities", "what can this do",
		"what am i allowed to type", "what can i type", "what can i type here", "what should i type", "whats the tour button do", "what are my options here",
		"help", "?", "menu", "commands", "what can you do", "what can i say",
		"what can i ask", "what do you do", "show commands", "list commands",
	},
	"history": {
		"show me our conversation", "our conversation", "the conversation", "what have we talked about", "show me what we said",
		"/history", "history", "show history", "what did i ask", "session history",
		"show the transcript", "our conversation so far",
	},
	"forget": {
		"/forget", "forget", "forget this session", "start over", "clear memory",
		"wipe the session", "reset the conversation",
	},
	"profile": {
		"/profile", "profile", "my profile", "show my profile", "what have you learned about me",
		"what do you know about me", "show what you learned",
	},
	"learn_style": {
		"/learn-style", "learn style", "learn my style", "learn my code style",
		"scan my style", "learn my conventions",
	},
	"predictor": {
		"/predictor", "predictor", "the predictor", "action predictor",
		"next action predictor", "how many actions logged",
	},
	"quit": {
		"quit", "exit", "stop", "bye", "goodbye", "close", "leave", "done", "im done",
		"/quit", "end session",
	},
	"run_make": {
		"do the {target} target", "do {target}", "the {target} target", "run target {target}",
		"run make {target}", "make {target}", "run the {target} target", "do make {target}",
		"execute make {target}", "make {target} please", "run {target}",
	},
	"explain_make": {
		"explain make {target}", "what does make {target} do", "explain the {target} target",
		"what is make {target}", "describe make {target}",
	},
	"concept_q": {
		"explain {concept} like im five", "explain {concept} to a beginner", "{concept} for beginners", "explain {concept} in plain words", "explain {concept} in simple terms", "i dont understand {concept}s", "help me understand {concept}s",
		"what is a {concept}", "what are {concept}s", "explain {concept} simply",
		"how do {concept}s work", "what does {concept} mean in go", "teach me {concept}",
		"{concept} explained", "go {concept} basics",
	},
	"write_code": {
		"write a function that reverses a string", "write a function to sort numbers",
		"write code for {concept}", "code a function that adds numbers",
		"write a helper that reads a file", "generate a function for {concept}",
		"write me a function", "write go code that prints hello",
	},
	"social": {
		"how was your day", "how is your day", "hows it going", "how are things", "whats up", "hows your day going", "did you have a good day", "you doing ok",
		"hello", "hi", "hey there", "good morning", "good evening", "how are you",
		"thanks", "thank you", "thats great", "nice", "cool", "tell me a joke",
		"who are you", "what are you", "you are helpful",
	},
	"why_func": {
		"why does {name} lock", "why does {name} return an error", "why does {name} exist",
		"why is {name} written like that", "how come {name} needs a lock", "why does {name} loop",
		"why does {name} iterate", "explain why {name} does that", "what is the reason {name} locks",
		"why does it lock", "why does this return an error", "why does it need a lock",
	},
	"compare": {
		"compare {name} and {name2}", "compare {name} with {name2}", "{name} vs {name2}",
		"difference between {name} and {name2}", "what is the difference between {name} and {name2}",
		"how is {name} different from {name2}", "how does {name} compare to {name2}",
		"{name} versus {name2}", "side by side {name} and {name2}",
	},
	"brains": {
		"what brains do you have", "show me your brains", "list your brains", "show brains",
		"what brains are in there", "tell me your brains", "brains", "the brains",
		"how are your brains doing", "brain status", "are your brains working",
		"how are the brains doing", "brain scoreboard", "which brains do you have",
	},
	"why_brain": {
		"why did you do that", "why did you answer that way", "who answered that",
		"which brain was that", "why that brain", "why did you use that brain",
		"who said that", "why did you say that", "why did you answer it that way",
		"which brain answered that", "why did you pick that brain",
	},
	"suggest": {
		"any suggestions", "what do you suggest", "any tips", "do you have any suggestions",
		"suggest something", "what would you suggest", "any advice", "give me a suggestion",
		"what do you recommend", "any recommendations", "suggestions please",
	},
	"start_guide": {
		"guide me", "start the guide", "get started", "im new here",
		"i am new here", "show me how this works", "guide me through this",
		"start guiding me", "take me through it", "walk me through everything",
		"guide me step by step", "take me step by step", "i just got here",
		"brand new here", "walk me through the whole thing",
	},
	"other": {
		"the weather is nice", "i like pizza", "what time is it in london",
		"buy me a coffee", "asdf qwer", "random words here", "my cat is cute",
		"stocks went up today", "i went to the store",
	},
	"tell_more": {
		"tell me more", "more", "tell me more about that", "tell me more about it",
		"go on", "continue", "keep going", "and then", "what else", "more info",
		"more about that", "elaborate", "expand on that", "dig deeper", "go deeper",
		"tell me more about {name}", "more about {name}", "what else does it do",
	},
	"what_now": {
		"what now", "what to do now", "what should i do now", "what next", "whats next",
		"what do i do", "what should i do", "im stuck", "im lost", "i dont know what to do",
		"help me", "how do i start", "where do i go from here", "what do i type",
		"how do i use this", "guide me", "directions please", "im confused",
		"i dont know what to type", "whats the next step", "next steps",
	
		"what is the next thing to do", "tell me the next thing to do", "what do i do next in this project",},
	"summarize": {
		"give me the summary", "give me a summary of this", "the summary of this codebase", "summary please", "short summary of this project",
		"what does this program do", "what does this app do", "what is this program",
		"what does this code do", "what does this project do", "what is this for",
		"summarize this project", "summary of this project", "what does it do",
		"describe this program", "project summary", "summarize this codebase",
		"in plain english what does this do", "what is this thing",
	},
}

var prefixes = []string{"", "", "", "please ", "can you ", "could you ", "hey ", "ok ", "so ", "pls "}
var suffixes = []string{"", "", "", "?", " please", " for me", " now"}

// typo introduces one beginner-style typo into a random word of len>=4.
func typo(s string, rng *rand.Rand) string {
	words := strings.Fields(s)
	var cand []int
	for i, w := range words {
		if len(w) >= 4 && !strings.ContainsAny(w, "/:.~") {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return s
	}
	i := cand[rng.Intn(len(cand))]
	w := words[i]
	pos := rng.Intn(len(w) - 1)
	switch rng.Intn(3) {
	case 0: // drop a letter
		w = w[:pos] + w[pos+1:]
	case 1: // swap neighbors
		if pos+1 < len(w) {
			w = w[:pos] + string(w[pos+1]) + string(w[pos]) + w[pos+2:]
		}
	case 2: // double a letter
		w = w[:pos] + string(w[pos]) + w[pos:]
	}
	words[i] = w
	return strings.Join(words, " ")
}

func fill(t string, rng *rand.Rand) string {
	for key, vals := range slots {
		for strings.Contains(t, "{"+key+"}") {
			t = strings.Replace(t, "{"+key+"}", vals[rng.Intn(len(vals))], 1)
		}
	}
	return t
}

func main() {
	rng := rand.New(rand.NewSource(7))
	seen := map[string]bool{}
	var out []pair
	add := func(input, intent string) {
		input = strings.TrimSpace(input)
		if input == "" || seen[input] {
			return
		}
		seen[input] = true
		out = append(out, pair{input, intent})
	}
	for intent, temps := range templates {
		for _, t := range temps {
			base := fill(t, rng)
			add(base, intent)
			for v := 0; v < 11; v++ {
				s := prefixes[rng.Intn(len(prefixes))] + fill(t, rng) + suffixes[rng.Intn(len(suffixes))]
				add(s, intent)
				if rng.Float64() < 0.35 {
					add(typo(s, rng), intent)
				}
			}
		}
	}
	f, err := os.Create("data/training/intent_pairs.jsonl")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	counts := map[string]int{}
	for _, p := range out {
		if err := enc.Encode(p); err != nil {
			panic(err)
		}
		counts[p.Intent]++
	}
	fmt.Printf("wrote %d pairs across %d intents\n", len(out), len(counts))
}
