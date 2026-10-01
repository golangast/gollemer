package chat

import "testing"

func TestGocliFalsePositives(t *testing.T) {
	cases := map[string]string{
		"build a web server":        "gocode",
		"check the weather":         "gocode", // pre-existing: gocodeCodegen ^check, out of scope,
		"can you check my spelling": "social",
		"what does go mod tidy do":  "go",
		"go mod tidy":               "gocli",
		"run go mod tidy for me":    "gocli",
		"I want to build my own PC": "go", // pre-existing: goTerms "build", out of scope
		"test the waters":           "social",
		"format my hard drive":      "social",
		"clean my room":             "social",
		"list my chores":            "social",
		"install a ceiling fan":     "social",
		"how do I run the tests":    "gocli",
		"what is the difference between go run and go build": "go",
		"how do I add two numbers in Go":                     "go", // pre-existing: goTerms, out of scope
		"write a function that sorts a list":                 "gocode",
		"are my deps up to date":                             "gocli",
		"what command formats my code":                      "gocli",
		"which command formats my code":                     "gocli",
	}
	for in, want := range cases {
		if got := routeDomain(in); got != want {
			t.Errorf("routeDomain(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestIsRunnableGoCommand(t *testing.T) {
	runnable := []string{
		"go mod tidy",
		"go build ./...",
		"gofmt -w .",
		"go test -race ./...",
		"go get -u github.com/google/uuid",
		"go doc fmt.Println",
		"  go version  ",
	}
	for _, cmd := range runnable {
		if !isRunnableGoCommand(cmd) {
			t.Errorf("isRunnableGoCommand(%q) = false, want true", cmd)
		}
	}
	blocked := []string{
		"",
		"howdy! ready to hear!",
		"go mod tidy; rm -rf /",
		"go build ./... && echo pwned",
		"go test | tee out.txt",
		"go vet > results.txt",
		"go env $(whoami)",
		"go list `date`",
		"rm -rf /",
		"python3 -c 'import os'",
		"echo go version",
		"ago mod tidy",
		"go;mod;tidy",
	}
	for _, cmd := range blocked {
		if isRunnableGoCommand(cmd) {
			t.Errorf("isRunnableGoCommand(%q) = true, want false", cmd)
		}
	}
}
