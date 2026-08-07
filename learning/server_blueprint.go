package learning

import (
	"fmt"
	"net/http"
)

func StartServer() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello from Gollemer Server!")
	})
	http.ListenAndServe(":8080", nil)
}
