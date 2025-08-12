package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

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

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "9000"
	}
	http.HandleFunc("/healthz", healthz)
	http.HandleFunc("/readyz", readyz)
	http.Handle("/metrics", promhttp.Handler())
	log.Printf("api-gateway listening on :%s", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}