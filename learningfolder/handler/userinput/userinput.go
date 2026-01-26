package userinput

import (
	"fmt"
	"net/http"
)

func UserInput(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "User Input Received (POST)")
}
