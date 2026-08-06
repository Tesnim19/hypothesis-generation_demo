"""Socket.IO / WebSocket API documentation (embedded in OpenAPI description)."""


def build_websocket_api_docs(ws_origin: str) -> str:
    """Return markdown for the real-time API section (not part of OpenAPI paths)."""
    return f"""\
## WebSocket / Real-Time API

The hypothesis service uses **Socket.IO** over WebSocket for real-time job progress.
Transport is **websocket only** (no long-polling fallback).

> **Note:** Socket.IO events are not part of the OpenAPI path list below. This section
> documents the full real-time contract.

### Connection

**Endpoint:** `{ws_origin}/`

The Socket.IO server is mounted at the root of the ASGI app (alongside FastAPI).
Connect with any Socket.IO client library.

**Authentication** — pass your JWT on connect (same token as REST `Authorization: Bearer`):

```javascript
import {{ io }} from "socket.io-client";

const socket = io("{ws_origin}", {{
  transports: ["websocket"],
  auth: {{ token: "<JWT>" }},
}});
```

Alternatives accepted at handshake:

- `Authorization: Bearer <JWT>` header
- Query string: `?token=<JWT>`

Connections without a valid token are **rejected**. Prefect worker tokens may include
`service: "prefect"` for internal relay only (not for browser clients).

**Requirements:** `REDIS_URL` must be set on the API and workers; progress events are
published via Redis pub/sub and relayed to subscribed rooms.

---

### Client → Server Events

#### `connect`

Emitted automatically by the Socket.IO client. The server validates JWT and accepts or
rejects the connection. There is no separate welcome `message` event.

#### `subscribe_hypothesis`

Subscribe to progress for a hypothesis job. Call after `POST /enrich` (202) or when
tracking an in-flight `POST /hypothesis` run.

**Payload:**

```json
{{ "hypothesis_id": "<hypothesis_id>" }}
```

**Behavior:**

- Joins room `hypothesis_<hypothesis_id>`
- Immediately emits one **`task_update`** to the caller with current state (cache replay)
- Returns ack: `{{ "status": "subscribed", "room": "hypothesis_<id>" }}` or `{{ "error": "..." }}`

If the hypothesis is already complete (`enrich_id`, `go_id`, `summary`, `graph` present),
the replayed event includes `status: "Completed"`, `progress: 100`, and `result`.

If tissue/LDSC data is partially available, replay may include:

```json
{{
  "tissue_rankings": [...],
  "tissue_results_ready": true,
  "causal_gene": "...",
  "enrichment_stage": "..."
}}
```

#### `subscribe_analysis`

Subscribe to GWAS analysis pipeline progress. Call after `POST /analysis-pipeline` (202).

**Payload:**

```json
{{ "project_id": "<project_id>" }}
```

**Behavior:**

- Requires authenticated user (not service token)
- Joins room `analysis_<project_id>`
- Immediately emits one **`analysis_update`** with saved pipeline state
- Returns ack: `{{ "status": "subscribed", "room": "analysis_<id>" }}`

#### `task_update` (service only)

Used internally by Prefect workers to relay updates through Redis. **Browser clients must
not emit this event** — it is ignored unless the session has `service: "prefect"`.

---

### Server → Client Events

#### `task_update`

Primary event for enrichment + hypothesis pipeline progress. Delivered to room
`hypothesis_<hypothesis_id>`.

**Payload structure:**

```json
{{
  "hypothesis_id": "<hypothesis_id>",
  "timestamp": "2026-08-06T12:00:00.000Z",
  "task": "<current_task_name>",
  "status": "Running",
  "progress": 42.5,
  "task_history": [
    {{ "timestamp": "...", "task": "...", "progress": 10 }}
  ]
}}
```

| Field | Description |
|-------|-------------|
| `status` | `Running`, `Completed`, or `Failed` |
| `progress` | 0–100 (enrichment ≈ 0–80%, hypothesis ≈ 80–100%) |
| `task` | Human-readable step name (see pipeline tables below) |
| `task_history` | Up to 5 most recently **completed** steps (no internal `state` field) |
| `next_task` | Optional hint for the upcoming step |
| `error` | Present when `status` is `Failed` |

**Subscribe replay (complete):**

```json
{{
  "hypothesis_id": "64a1f3c2-b0e4-412d-8c8f-9e0123456789",
  "status": "Completed",
  "progress": 100,
  "result": {{ "...": "full hypothesis document" }},
  "task_history": []
}}
```

**Subscribe replay (failed):**

```json
{{
  "hypothesis_id": "64a1f3c2-b0e4-412d-8c8f-9e0123456789",
  "status": "Failed",
  "progress": 35,
  "error": "Pipeline error message",
  "task_history": []
}}
```

#### `analysis_update`

GWAS analysis pipeline state. Delivered to room `analysis_<project_id>`.

**Payload structure:**

```json
{{
  "project_id": "<project_id>",
  "user_id": "<user_id>",
  "timestamp": "2026-08-06T12:00:00.000Z",
  "status": "Running",
  "stage": "Harmonization",
  "progress": 10,
  "message": "Starting Nextflow harmonization"
}}
```

| Field | Description |
|-------|-------------|
| `status` | `Running`, `Completed`, or `Failed` |
| `stage` | Pipeline stage identifier (see analysis stages table) |
| `progress` | Approximate 0–100 completion |
| `message` | Human-readable status line for UI |
| `ldsc_status` | Optional; set when LDSC/tissue step finishes |

**Completed example:**

```json
{{
  "project_id": "...",
  "status": "Completed",
  "stage": "LDSC_Tissue_Analysis",
  "progress": 100,
  "message": "Analysis completed successfully",
  "ldsc_status": "completed"
}}
```

**Failed example:**

```json
{{
  "project_id": "...",
  "status": "Failed",
  "stage": "Cojo",
  "progress": 50,
  "message": "COJO analysis failed - no independent signals found"
}}
```

---

### Task Status Values

| Status | Description |
|--------|-------------|
| `Running` | Job in progress — partial results may be available via REST or subscribe replay |
| `Completed` | Pipeline stage or full job finished successfully |
| `Failed` | Unrecoverable error; see `error` or `message` |

Internal task `state` values (`started`, `completed`, `failed`, `retrying`) are tracked
server-side but stripped from public `task_history` entries.

---

### Enrichment Pipeline Stages (`task_update`)

Triggered by `POST /enrich` → Prefect enrichment flow. Progress weights total **80%**
of the combined bar.

| Task name | Phase |
|-----------|-------|
| Verifying existence of enrichment data | Cache check |
| Getting candidate genes | Variant → gene mapping |
| Predicting causal gene | Causal gene inference |
| Getting relevant gene proof | Knowledge-graph proof graphs |
| Retrying to predict causal gene | Retry path (state: retrying) |
| Retrying to get relevant gene proof | Retry path |
| Creating enrich data | Persist enrichment + tissue metadata |
| Enrichment Analysis (N/M) | Per-graph tissue enrichment (optional) |

`status: "Completed"` with `progress: 100` is emitted when **Creating enrich data**
finishes (enrichment-only jobs). Hypothesis generation adds the remaining 20%.

---

### Hypothesis Pipeline Stages (`task_update`)

Triggered by `POST /hypothesis` (sync REST, but emits live updates if subscribed).
Progress weights total **20%** of the combined bar.

| Task name | Phase |
|-----------|-------|
| Verifying existence of hypothesis data | Cache check |
| Getting enrichement data | Load enrichment record |
| Getting gene data | Gene / coexpression lookup |
| Querying gene data | Prolog gene query |
| Querying variant data | Prolog variant query |
| Querying phenotype data | Prolog phenotype query |
| Generating graph summary | LLM graph summarization |
| Generating hypothesis | Persist summary + graph |

`status: "Completed"` with `progress: 100` when **Generating hypothesis** completes.

---

### Analysis Pipeline Stages (`analysis_update`)

Triggered by `POST /analysis-pipeline` → Prefect analysis flow.

| Stage | Typical message |
|-------|-----------------|
| `File_upload` | Uploading or preparing GWAS file... |
| `Harmonization` | Starting Nextflow harmonization |
| `Filtering` | Harmonization completed, filtering significant variants |
| `Cojo` | Filtering completed, running COJO analysis |
| `Fine_mapping` | COJO analysis completed, starting fine-mapping |
| `LDSC_Tissue_Analysis` | Fine-mapping completed, waiting for LDSC and tissue analysis |
| `LDSC_Analysis` | LDSC failure path (status may be `Failed`) |

---

### Typical Client Flows

#### Enrichment → hypothesis graph

```
1. POST /enrich                    → 202 {{ hypothesis_id, project_id }}
2. socket.emit("subscribe_hypothesis", {{ hypothesis_id }})
3. socket.on("task_update", handler)
   → {{ status: "Running", task: "Getting candidate genes", progress: 12.5 }}
   → {{ status: "Running", task: "Creating enrich data", progress: 75 }}
   → {{ status: "Completed", progress: 100 }}   // enrichment done
4. POST /hypothesis?id=<enrich_id>&go=<go_term>  → summary + graph (or subscribe during run)
5. GET /hypothesis?id=<hypothesis_id>            → full result if needed
```

#### GWAS analysis pipeline

```
1. POST /analysis-pipeline         → 202 {{ project_id, status: "started" }}
2. socket.emit("subscribe_analysis", {{ project_id }})
3. socket.on("analysis_update", handler)
   → {{ stage: "Harmonization", status: "Running", progress: 10 }}
   → {{ stage: "Fine_mapping", status: "Running", progress: 70 }}
   → {{ status: "Completed", progress: 100, message: "Analysis completed successfully" }}
4. GET /projects?id=<project_id>   → project with analysis metadata
5. GET /credible-sets?project_id=... → fine-mapping outputs
```
"""
