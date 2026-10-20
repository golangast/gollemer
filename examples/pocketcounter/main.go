// Pocketcounter is the tiny example project bundled with Gollemer.
// The chat's guided tour uses it when no real project has been
// analyzed yet, so a beginner can see every command work in the
// first minute — no clone, no setup.
package main

import "fmt"

func main() {
	c := NewCounter()
	c.Add("apples")
	c.Add("apples")
	c.Add("pears")

	apples, _ := c.Get("apples")
	fmt.Println("apples:", apples)
	fmt.Println("total:", c.Total())

	if _, err := c.Get("oranges"); err != nil {
		fmt.Println("oranges:", err)
	}
}
