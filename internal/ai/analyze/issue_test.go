package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// issueFixture mirrors the updatecli shape: a ResourceConfig struct whose
// fields match a YAML example, an inlining source.Config, and a shared
// httpclient package.
func issueFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/iss\n\ngo 1.21\n")
	write("main.go", "package main\n\nfunc main() { run() }\n\nfunc run() { fetch(\"http://x\") }\n")
	write("resource/resource.go", `package resource

// ResourceConfig defines a resource configuration.
type ResourceConfig struct {
	// Name specifies the resource name
	Name string `+"`yaml:\",omitempty\"`"+`
	// Kind specifies the resource kind
	Kind string `+"`yaml:\",omitempty\"`"+`
	// Spec defines the resource spec
	Spec interface{} `+"`yaml:\",omitempty\"`"+`
}
`)
	write("source/source.go", `package source

import "example.com/iss/resource"

// Config defines a source configuration.
type Config struct {
	resource.ResourceConfig `+"`yaml:\",inline\"`"+`
}
`)
	write("httpclient/httpclient.go", `package httpclient

// HTTPClient is the contract of ALL http clients used by sources.
type HTTPClient interface {
	Do(req string) (string, error)
}

// New builds the shared client.
func New() HTTPClient { return nil }
`)
	write("httpclient/retry.go", `package httpclient

// RetryClient wraps Do with retry+backoff.
type RetryClient struct{}

// NewRetryClient builds the shared retrying client.
func NewRetryClient() *RetryClient { return nil }
`)
	write("source/fetch.go", `package source

import "example.com/iss/httpclient"

func fetch(url string) string {
	c := httpclient.New()
	s, _ := c.Do(url)
	return s
}
`)
	return root
}

const issueText = `Feature Request: retry+backoff mechanism

When I use URLs to fetch files if the endpoint is flaky then the whole pipeline fails.

I'd like the support for retry+backoff for all the sources that support URLs.

` + "```" + `
sources:
  ubi_version:
    kind: yaml
    spec:
      file: 'https://example.com/x.yaml'
    retry:
      limit: 3
      delay: 3
` + "```"

// The NLP pulls the config keys, the mechanism, and the action.
func TestExtractIssueConcepts(t *testing.T) {
	c := ExtractIssueConcepts(issueText)
	if !strings.Contains(c.Title, "retry") {
		t.Errorf("title = %q", c.Title)
	}
	for _, want := range []string{"kind", "spec", "retry", "limit", "delay"} {
		found := false
		for _, k := range c.YamlKeys {
			if k == want {
				found = true
			}
		}
		if !found {
			t.Errorf("YamlKeys missing %q: %v", want, c.YamlKeys)
		}
	}
	if len(c.Mechanisms) == 0 {
		t.Error("no mechanisms extracted")
	}
	has := func(ss []string, w string) bool {
		for _, s := range ss {
			if s == w {
				return true
			}
		}
		return false
	}
	if !has(c.Mechanisms, "url") {
		t.Errorf("mechanisms = %v, want url", c.Mechanisms)
	}
	if c.Action != "add" {
		t.Errorf("action = %q, want add", c.Action)
	}
}

// The YAML keys find the config struct, including through embedding.
func TestFindConfigStruct(t *testing.T) {
	p, err := Analyze(issueFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c := ExtractIssueConcepts(issueText)
	matches := p.FindConfigStruct(c.YamlKeys)
	if len(matches) == 0 {
		t.Fatal("no config struct found")
	}
	top := matches[0]
	// source.Config inlines ResourceConfig: the source-level struct is the
	// beginner-friendly answer, and the inlined one must also match.
	if top.Type.Name != "Config" {
		t.Errorf("top match = %s, want Config (the source-level struct)", top.Type.Name)
	}
	seenResource := false
	for _, m := range matches {
		if m.Type.Name == "ResourceConfig" {
			seenResource = true
		}
	}
	if !seenResource {
		t.Errorf("ResourceConfig not among matches: %v", matches)
	}
	for _, want := range []string{"kind", "spec"} {
		found := false
		for _, h := range top.Hits {
			if h == want {
				found = true
			}
		}
		if !found {
			t.Errorf("hits = %v, want %q", top.Hits, want)
		}
	}
	// retry/limit/delay are the proposed new keys.
	has := func(ss []string, w string) bool {
		for _, s := range ss {
			if s == w {
				return true
			}
		}
		return false
	}
	if !has(top.New, "retry") || !has(top.New, "limit") {
		t.Errorf("new keys = %v, want retry and limit", top.New)
	}
}

// The mechanism word finds the shared package.
func TestFindMechanism(t *testing.T) {
	p, err := Analyze(issueFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	site := p.FindMechanism("http")
	if site == nil {
		t.Fatal("no mechanism site for http")
	}
	if site.Pkg.Name != "httpclient" {
		t.Errorf("site = %s, want httpclient", site.Pkg.Name)
	}
}

// The full guided answer names the config struct, the mechanism
// package, and the call chain.
func TestGuideIssue(t *testing.T) {
	p, err := Analyze(issueFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := p.GuideIssue(ExtractIssueConcepts(issueText))
	for _, want := range []string{
		"TO ADD",
		"CONFIG",
		"type Config struct",
		"BEHAVIOR",
		"httpclient",
		"YOU ARE HERE:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("GuideIssue missing %q\n%s", want, out)
		}
	}
	t.Logf("\n%s", out)
}

// Indentation separates struct-level additions from nested keys:
// retry: is new at kind:'s level, limit:/delay: nest under it,
// file: nests under the existing spec: and sources: is a wrapper.
func TestNewKeysAtLevel(t *testing.T) {
	p, err := Analyze(issueFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c := ExtractIssueConcepts(issueText)
	matches := p.FindConfigStruct(c.YamlKeys)
	if len(matches) == 0 {
		t.Fatal("no config struct found")
	}
	add, nested := newKeysAtLevel(c, matches[0])
	if len(add) != 1 || add[0] != "retry" {
		t.Errorf("add = %v, want [retry]", add)
	}
	kids := nested["retry"]
	has := func(ss []string, w string) bool {
		for _, s := range ss {
			if s == w {
				return true
			}
		}
		return false
	}
	if !has(kids, "limit") || !has(kids, "delay") {
		t.Errorf("nested[retry] = %v, want limit and delay", kids)
	}
	out := p.GuideIssue(c)
	if !strings.Contains(out, "Add the issue's new key here: `retry`") {
		t.Errorf("missing precise add line\n%s", out)
	}
	if !strings.Contains(out, "with `limit`, `delay` underneath") {
		t.Errorf("missing nested keys line\n%s", out)
	}
}
