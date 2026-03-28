// Main implementation for the file object
package main

import "fmt"

func main() {
	http.HandleFunc("/jim", WithHandler)
	// HANDLER_REGISTRATIONS_GO_HERE
	fmt.Println("Executing main (file logic)")
}
