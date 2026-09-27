package seq2seq

import "testing"

func TestNormalizeQueryStripsPunct(t *testing.T) {
	a := NormalizeQuery("Tell me a joke.")
	b := NormalizeQuery("tell me a joke")
	c := NormalizeQuery("  Tell   me a joke!  ")
	if a != b || b != c {
		t.Fatalf("not canonical: %q %q %q", a, b, c)
	}
	if NormalizeQuery("I'm happy!") != "i'm happy" {
		t.Fatalf("apostrophe broken: %q", NormalizeQuery("I'm happy!"))
	}
}
