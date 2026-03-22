package home

import (
	"net/http"

	"mm/pkg/render"
)

func Home(w http.ResponseWriter, r *http.Request) {
	render.Template(w, "home.html", map[string]string{"Title": "Home Page"})
}
