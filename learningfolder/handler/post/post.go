package post

import (
	"fmt"
	"net/http"
)

func Posts(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Posts Page (POST)")
}
