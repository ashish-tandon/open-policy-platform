package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

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

func envOrDefault(env, def string) string {
	v := os.Getenv(env)
	if v == "" {
		return def
	}
	return v
}

func serviceMap() map[string]string {
	return map[string]string{
		"auth-service":          envOrDefault("AUTH_SERVICE_URL", "http://auth-service:9001"),
		"policy-service":        envOrDefault("POLICY_SERVICE_URL", "http://policy-service:9002"),
		"search-service":        envOrDefault("SEARCH_SERVICE_URL", "http://search-service:9003"),
		"notification-service":  envOrDefault("NOTIF_SERVICE_URL", "http://notification-service:9004"),
		"config-service":        envOrDefault("CONFIG_SERVICE_URL", "http://config-service:9005"),
		"monitoring-service":    envOrDefault("MONITORING_SERVICE_URL", "http://monitoring-service:9006"),
		"etl":                   envOrDefault("ETL_SERVICE_URL", "http://etl:9007"),
		"scraper-service":       envOrDefault("SCRAPER_SERVICE_URL", "http://scraper-service:9008"),
		"mobile-api":            envOrDefault("MOBILE_API_URL", "http://mobile-api:9009"),
		"legacy-django":         envOrDefault("LEGACY_DJANGO_URL", "http://legacy-django:9010"),
	}
}

func makeReverseProxy(target string) *httputil.ReverseProxy {
	u, _ := url.Parse(target)
	return httputil.NewSingleHostReverseProxy(u)
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Second}
	statuses := map[string]any{}
	for name, base := range serviceMap() {
		url := strings.TrimRight(base, "/") + "/healthz"
		st := map[string]any{"status": "unknown", "target": base}
		resp, err := client.Get(url)
		if err == nil {
			st["http"] = resp.StatusCode
			if resp.StatusCode == 200 {
				st["status"] = "ok"
			} else {
				st["status"] = "error"
			}
		} else {
			st["error"] = err.Error()
			st["status"] = "error"
		}
		statuses[name] = st
	}
	json.NewEncoder(w).Encode(statuses)
}

func gatewayHandler(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	// Map prefixes to service URLs (env-configurable; K8s DNS defaults)
	routes := map[string]string{
		"/api/auth/":          envOrDefault("AUTH_SERVICE_URL", "http://auth-service:9001"),
		"/api/policies/":      envOrDefault("POLICY_SERVICE_URL", "http://policy-service:9002"),
		"/api/committees/":    envOrDefault("POLICY_SERVICE_URL", "http://policy-service:9002"),
		"/api/debates/":       envOrDefault("POLICY_SERVICE_URL", "http://policy-service:9002"),
		"/api/votes/":         envOrDefault("POLICY_SERVICE_URL", "http://policy-service:9002"),
		"/api/search/":        envOrDefault("SEARCH_SERVICE_URL", "http://search-service:9003"),
		"/api/notifications/": envOrDefault("NOTIF_SERVICE_URL", "http://notification-service:9004"),
		"/api/config/":        envOrDefault("CONFIG_SERVICE_URL", "http://config-service:9005"),
		"/api/monitoring/":    envOrDefault("MONITORING_SERVICE_URL", "http://monitoring-service:9006"),
		"/api/etl/":           envOrDefault("ETL_SERVICE_URL", "http://etl:9007"),
		"/api/scrapers/":      envOrDefault("SCRAPER_SERVICE_URL", "http://scraper-service:9008"),
		"/api/mobile/":        envOrDefault("MOBILE_API_URL", "http://mobile-api:9009"),
		"/api/legacy/":        envOrDefault("LEGACY_DJANGO_URL", "http://legacy-django:9010"),
	}
	if strings.HasPrefix(path, "/api/status") {
		statusHandler(w, r)
		return
	}
	for prefix, target := range routes {
		if strings.HasPrefix(path, prefix) {
			makeReverseProxy(target).ServeHTTP(w, r)
			return
		}
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