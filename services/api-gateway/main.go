package main

import (
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"

	promhttp "github.com/prometheus/client_golang/prometheus/promhttp"
)

func healthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func readyz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func makeReverseProxy(target string) *httputil.ReverseProxy {
	u, _ := url.Parse(target)
	return httputil.NewSingleHostReverseProxy(u)
}

func gatewayHandler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if strings.HasPrefix(path, "/api/auth/") {
		makeReverseProxy("http://localhost:9001").ServeHTTP(w, r)
		return
	}
	if strings.HasPrefix(path, "/api/policies/") || strings.HasPrefix(path, "/api/committees/") || strings.HasPrefix(path, "/api/debates/") || strings.HasPrefix(path, "/api/votes/") {
		makeReverseProxy("http://localhost:9002").ServeHTTP(w, r)
		return
	}
	http.NotFound(w, r)
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "9000"
	}
	http.HandleFunc("/healthz", healthz)
	http.HandleFunc("/readyz", readyz)
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/", gatewayHandler)
	log.Printf("api-gateway listening on :%s", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}