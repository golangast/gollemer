package render

import (
	"html/template"
	"net/http"
)

var Tmpl *template.Template

func Template(w http.ResponseWriter, name string, data interface{}) {
	if Tmpl == nil {
		http.Error(w, "Templates not initialized", http.StatusInternalServerError)
		return
	}
	err := Tmpl.ExecuteTemplate(w, name, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
