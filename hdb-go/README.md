# hdb-go — HydraDB row API

Serves the four table-scoped routes consumed by `researcher/mutator.py`.
Wire-compatible with the Python reference in `research_store/app.py`.

## Routes

| Method | Path | Purpose |
|--------|------|---------|
| POST   | `/tables/{table}/rows`      | insert one record (body: JSON object) |
| GET    | `/tables/{table}/rows`      | list records (optional `?status=`) |
| GET    | `/tables/{table}/rows/{id}` | get single record by id |
| DELETE | `/tables/{table}/rows/{id}` | delete record by id |

Every route requires:
- **Auth** — `Authorization: Bearer <key>` matching env `HYDRA_DB_API_KEY`.
- **Tenant** — `X-Tenant-ID` and `X-Sub-Tenant-ID` headers. Records are stored
  and scoped under that `(tenant, sub_tenant)` pair. `id` (UUID) and `created_at`
  (RFC3339) are server-generated.

## Build & run

```sh
cd hdb-go
go mod tidy          # fetches github.com/google/uuid
HYDRA_DB_API_KEY=sk_live_... PORT=8000 go run .
```

Requires Go 1.22+ (method + wildcard `ServeMux` patterns).

> Note: the store here is in-memory (process-local). For production at
> api.hydradb.com, replace the `store` struct with your real datastore;
> the handler/auth/tenant logic is unchanged.

## Verify

```sh
curl -fsS -X POST localhost:8000/tables/strategy_candidates/rows \
  -H "Authorization: Bearer $HYDRA_DB_API_KEY" \
  -H "X-Tenant-ID: agents" -H "X-Sub-Tenant-ID: day_trading" \
  -H "Content-Type: application/json" \
  -d '{"name":"x","status":"candidate","estimated_sharpe":1.4,"data_requirements":["1m_bars"]}'
```
