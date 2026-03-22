package createuser

import (
	"net/http"

	"mm/pkg/render"
)

func Createuser(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		render.Template(w, "form.html", nil)
		return
	}
	// POST case
	render.Template(w, "form.html", map[string]bool{"Success": true})
}
