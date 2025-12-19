package persons
import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	_ "modernc.org/sqlite"
)

type Persons struct {
	ID int `json:"id"`
	Name string `json:"name"`
}



func UpdatePersonsHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /update/user/123
		http.Error(w, "Invalid URL, expecting /update/persons/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	var u Persons
	err := json.NewDecoder(r.Body).Decode(&u)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	db, err := sql.Open("sqlite", "persons.db")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("UPDATE persons SET name = ? WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = stmt.Exec(u.Name, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%!s(MISSING) with ID %s updated successfully", id)
}

func DeletePersonsHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /delete/user/123
		http.Error(w, "Invalid URL, expecting /delete/persons/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	db, err := sql.Open("sqlite", "persons.db")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("DELETE FROM persons WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_, err = stmt.Exec(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%!s(MISSING) with ID %s deleted successfully", id)
}
