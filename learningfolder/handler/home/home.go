package home

import (
	"net/http"

	"github.com/golangast/gollemer/learningfolder/pkg/render"
)

func Home(w http.ResponseWriter, r *http.Request) {
	render.Template(w, "home.html", map[string]string{"Title": "Home Page"})
}
