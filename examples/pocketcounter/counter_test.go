package main

import "testing"

func TestCounterTotal(t *testing.T) {
	c := NewCounter()
	c.Add("apples")
	c.Add("apples")
	c.Add("pears")
	if got := c.Total(); got != 3 {
		t.Fatalf("Total() = %d, want 3", got)
	}
	if _, err := c.Get("oranges"); err == nil {
		t.Fatal("Get(oranges) should fail: it was never counted")
	}
}
