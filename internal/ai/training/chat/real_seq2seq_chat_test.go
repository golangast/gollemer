package chat

import "testing"

// The chatter detector is the explicit mode-separation boundary: anything it
// matches is answered by the chat model, everything else goes to the code
// decoder. It must catch common openers while never stealing code requests.
func TestChatterPromptRouting(t *testing.T) {
	chatter := []string{
		"hello", "hello!", "hi", "hey",
		"good morning", "good evening",
		"how are you", "how are you?",
		"thanks", "thank you",
		"who are you", "tell me a joke",
	}
	for _, q := range chatter {
		if !chatterPrompt.MatchString(q) {
			t.Errorf("chatter %q not detected; would reach the code decoder", q)
		}
	}
	code := []string{
		"write a function that adds two ints",
		"how do i check even numbers in go",
		"code me an adder for integers",
		"write a hello world program in go",
		"i need a rect type with width and height",
		"how do i sum a slice in go",
		// greeting-prefixed code requests stay on the code model
		"hey, write a function that adds two ints",
		"hello, can you write me a factorial function",
	}
	for _, q := range code {
		if chatterPrompt.MatchString(q) && !gocodeTerms.MatchString(q) {
			t.Errorf("code request %q misrouted to chatter", q)
		}
	}
}
