package users
import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

type Users struct {
	ID int `json:"id"`
	Age int `json:"age"`
	Name string `json:"name"`
}



func ShowUsersHandler(w http.ResponseWriter, r *http.Request) {
	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "jim", "cmd", "jim", "users", "users.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	rows, err := db.Query("SELECT id, age, name FROM users")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	results := make([]Users, 0)
	for rows.Next() {
		var u Users
		if err := rows.Scan(&u.ID, &u.Age, &u.Name); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		results = append(results, u)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func UpdateUsersHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /update/user/123
		http.Error(w, "Invalid URL, expecting /update/users/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	var u Users
	err := json.NewDecoder(r.Body).Decode(&u)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "jim", "cmd", "jim", "users", "users.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("UPDATE users SET age = ?, name = ? WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = stmt.Exec(u.Age, u.Name, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%!s(MISSING) with ID %s updated successfully", id)
}

func DeleteUsersHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /delete/user/123
		http.Error(w, "Invalid URL, expecting /delete/users/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "jim", "cmd", "jim", "users", "users.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("DELETE FROM users WHERE id = ?")
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

func CreateUsersHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseForm()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	name := r.FormValue("name")
	age := r.FormValue("age")

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "jim", "cmd", "jim", "users", "users.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("INSERT INTO users (name, age) VALUES (?, ?)")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = stmt.Exec(name, age)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

    fmt.Fprintf(w, "User %s created successfully", name)
}

func CreateUsersFormHandler(w http.ResponseWriter, r *http.Request) {
	form := `
<!DOCTYPE html>
<html>
<head>
<title>Create User</title>
</head>
<body>
<h1>Create User</h1>
<form action="/create/users/" method="post">
<label for="name">Name:</label><br>
<input type="text" id="name" name="name"><br>
<label for="age">Age:</label><br>
<input type="text" id="age" name="age"><br><br>
<input type="submit" value="Submit">
</form>
</body>
</html>
`
	fmt.Fprint(w, form)
}
