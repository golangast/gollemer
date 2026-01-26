package routes

import (
	"net/http"

	"github.com/golangast/gollemer/learningfolder/handler/createuser"
	"github.com/golangast/gollemer/learningfolder/handler/home"
	"github.com/golangast/gollemer/learningfolder/handler/loginemail"
	"github.com/golangast/gollemer/learningfolder/handler/post"
	"github.com/golangast/gollemer/learningfolder/handler/profile"
	"github.com/golangast/gollemer/learningfolder/handler/userinput"
)

func RegisterRoutes(mux *http.ServeMux) {
	// GET
	mux.HandleFunc("GET /", home.Home)
	mux.HandleFunc("GET /usercreate", profile.Profile)
	mux.HandleFunc("GET /loginemail/{email}/{sitetoken}", loginemail.LoginEmail)

	// POST
	mux.HandleFunc("POST /usercreate", createuser.Createuser)
	mux.HandleFunc("POST /userinput", userinput.UserInput)
	mux.HandleFunc("POST /p", post.Posts)

	// WASM and Static Files with cache headers
	wasmHandler := cacheControlMiddleware(http.StripPrefix("/wasm/", http.FileServer(http.Dir("wasm"))), "public, max-age=31536000, immutable")
	mux.Handle("GET /wasm/", wasmHandler)

	assetsHandler := cacheControlMiddleware(http.StripPrefix("/assets/", http.FileServer(http.Dir("assets"))), "public, max-age=31536000, immutable")
	mux.Handle("GET /assets/", assetsHandler)
}

// cacheControlMiddleware adds Cache-Control headers to responses
func cacheControlMiddleware(next http.Handler, cacheControl string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", cacheControl)
		next.ServeHTTP(w, r)
	})
}
