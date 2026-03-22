package loginemail

import (
	"fmt"
	"net/http"
)

func LoginEmail(w http.ResponseWriter, r *http.Request) {
	email := r.PathValue("email")
	token := r.PathValue("sitetoken")
	fmt.Fprintf(w, "Login Email: %s, Token: %s", email, token)
}
