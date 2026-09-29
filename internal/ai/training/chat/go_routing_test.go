package chat

import "testing"

// Regression test for 2026-09-29: "what is go" and "what is a function"
// were routing to social (and getting song lyrics / mush). They must
// reach the Go concept domain, while plain social phrases with "go"
// in them must not.
func TestGoRoutingBasics(t *testing.T) {
	cases := map[string]string{
		"what is go":                                        "go",
		"what is a function":                                "go",
		"what is a goroutine":                               "go",
		"how do i print formatted text in go":               "go",
		"how do i split a string in go":                     "go",
		"what is the difference between Printf and Sprintf": "go",
		"can go use multiple cpu cores":                     "go",
		"how do i exit a go program with an error code":     "go",
		"lets go to the park":                               "social",
		"when should i wrap an error":                       "go",
		"what is a go workspace":                            "go",
	}
	for in, want := range cases {
		if got := routeDomain(in); got != want {
			t.Errorf("routeDomain(%q) = %s, want %s", in, got, want)
		}
	}
}
