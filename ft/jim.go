// Jim implementation for the webserver object
package ft

import (
	"fmt"
	"net/http"
)

func init() {
	fmt.Println("Initializing jim (webserver logic)")
}

func StartServer() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello from Gollemer Server!")
	})
	http.ListenAndServe(":8080", nil)
}

func startmyserver() {

	return
}

type named struct {
	name string
}

type jill  {
	cat string
	age int
}

func addJake() int {
	a := 1
	b := 2
	return a + b
}
type after struct {
}
