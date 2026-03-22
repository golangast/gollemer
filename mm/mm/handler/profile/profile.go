package profile

import (
	"fmt"
	"net/http"
)

func Profile(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Profile Page")
}
