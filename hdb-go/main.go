// Command hdb-go serves the HydraDB row API at api.hydradb.com.
//
// It exposes exactly four table-scoped routes:
//
//	POST   /tables/{table}/rows        insert one record (body: JSON object)
//	GET    /tables/{table}/rows        list records (optional ?status=)
//	GET    /tables/{table}/rows/{id}   get single record by id
//	DELETE /tables/{table}/rows/{id}   delete record by id
//
// Every route requires:
//   - Auth:   Authorization: Bearer <key>  matching env HYDRA_DB_API_KEY
//   - Tenant: X-Tenant-ID and X-Sub-Tenant-ID headers; records are stored and
//     scoped under that (tenant, sub_tenant) pair.
//
// This mirrors research_store/app.py (the Python reference) byte-for-byte on the
// wire. The store here is in-memory; swap store{} for a real DB in production.
//
// Requires Go 1.22+ (method-and-wildcard ServeMux patterns).
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

type record = map[string]any

// store is an in-memory, tenant-scoped row store keyed by
// (tenant, subTenant, table) -> id -> record.
type store struct {
	mu   sync.Mutex
	data map[string]map[string]record
}

func newStore() *store { return &store{data: map[string]map[string]record{}} }

func key(tenant, sub, table string) string { return tenant + "\x00" + sub + "\x00" + table }

func (s *store) bucket(tenant, sub, table string) map[string]record {
	k := key(tenant, sub, table)
	if s.data[k] == nil {
		s.data[k] = map[string]record{}
	}
	return s.data[k]
}

type server struct {
	apiKey string
	store  *store
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func errJSON(w http.ResponseWriter, code int, detail string) {
	writeJSON(w, code, map[string]string{"detail": detail})
}

// auth enforces the bearer token; tenant returns the (tenant, sub) pair or
// writes a 400 and reports ok=false.
func (s *server) authed(w http.ResponseWriter, r *http.Request) bool {
	if s.apiKey == "" {
		errJSON(w, http.StatusInternalServerError, "HYDRA_DB_API_KEY not configured on server")
		return false
	}
	if r.Header.Get("Authorization") != "Bearer "+s.apiKey {
		errJSON(w, http.StatusUnauthorized, "invalid or missing bearer token")
		return false
	}
	return true
}

func (s *server) tenant(w http.ResponseWriter, r *http.Request) (string, string, bool) {
	t := r.Header.Get("X-Tenant-ID")
	st := r.Header.Get("X-Sub-Tenant-ID")
	if t == "" || st == "" {
		errJSON(w, http.StatusBadRequest, "X-Tenant-ID and X-Sub-Tenant-ID headers are required")
		return "", "", false
	}
	return t, st, true
}

func (s *server) insertRow(w http.ResponseWriter, r *http.Request) {
	if !s.authed(w, r) {
		return
	}
	tenant, sub, ok := s.tenant(w, r)
	if !ok {
		return
	}
	var body record
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body == nil {
		errJSON(w, http.StatusBadRequest, "body must be a JSON object")
		return
	}
	table := r.PathValue("table")
	id := uuid.NewString()
	body["id"] = id // server-generated; any client id is overwritten
	if _, exists := body["created_at"]; !exists {
		body["created_at"] = time.Now().UTC().Format(time.RFC3339Nano)
	}
	body["tenant_id"] = tenant
	body["sub_tenant_id"] = sub

	s.store.mu.Lock()
	s.store.bucket(tenant, sub, table)[id] = body
	s.store.mu.Unlock()

	writeJSON(w, http.StatusCreated, body)
}

func (s *server) listRows(w http.ResponseWriter, r *http.Request) {
	if !s.authed(w, r) {
		return
	}
	tenant, sub, ok := s.tenant(w, r)
	if !ok {
		return
	}
	table := r.PathValue("table")
	statusFilter := r.URL.Query().Get("status")
	hasFilter := r.URL.Query().Has("status")

	s.store.mu.Lock()
	rows := make([]record, 0)
	for _, rec := range s.store.bucket(tenant, sub, table) {
		if hasFilter {
			if v, _ := rec["status"].(string); v != statusFilter {
				continue
			}
		}
		rows = append(rows, rec)
	}
	s.store.mu.Unlock()

	sort.Slice(rows, func(i, j int) bool {
		ci, _ := rows[i]["created_at"].(string)
		cj, _ := rows[j]["created_at"].(string)
		return ci < cj
	})
	writeJSON(w, http.StatusOK, map[string]any{"rows": rows, "count": len(rows)})
}

func (s *server) getRow(w http.ResponseWriter, r *http.Request) {
	if !s.authed(w, r) {
		return
	}
	tenant, sub, ok := s.tenant(w, r)
	if !ok {
		return
	}
	table, id := r.PathValue("table"), r.PathValue("id")
	s.store.mu.Lock()
	rec, found := s.store.bucket(tenant, sub, table)[id]
	s.store.mu.Unlock()
	if !found {
		errJSON(w, http.StatusNotFound, "row not found")
		return
	}
	writeJSON(w, http.StatusOK, rec)
}

func (s *server) deleteRow(w http.ResponseWriter, r *http.Request) {
	if !s.authed(w, r) {
		return
	}
	tenant, sub, ok := s.tenant(w, r)
	if !ok {
		return
	}
	table, id := r.PathValue("table"), r.PathValue("id")
	s.store.mu.Lock()
	b := s.store.bucket(tenant, sub, table)
	_, found := b[id]
	delete(b, id)
	s.store.mu.Unlock()
	if !found {
		errJSON(w, http.StatusNotFound, "row not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"deleted": id})
}

func main() {
	s := &server{apiKey: os.Getenv("HYDRA_DB_API_KEY"), store: newStore()}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /tables/{table}/rows", s.insertRow)
	mux.HandleFunc("GET /tables/{table}/rows", s.listRows)
	mux.HandleFunc("GET /tables/{table}/rows/{id}", s.getRow)
	mux.HandleFunc("DELETE /tables/{table}/rows/{id}", s.deleteRow)

	addr := ":" + envOr("PORT", "8000")
	log.Printf("hdb-go listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
