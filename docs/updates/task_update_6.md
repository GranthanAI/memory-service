# Task Update 6: Clients, Security, Startup Validation, and Graceful Shutdown

This update documents the technical implementation details for the following phases in the GraphGPT Memory Service:
1. **Phase 23: Graph Service Client** (Shared AsyncClient pool, Circuit Breaker, Retries with Exponential Backoff).
2. **Phase 24: Embedding Client Abstraction** (Abstract Interface, GRPCEmbeddingClient, MockEmbeddingClient, Config integration).
3. **Phase 25: Service-to-Service Security** (Dependency-free HS256 JWT, API Key verification middleware).
4. **Phase 26: Startup Validation** (Cassandra schemas, Redis ping, Milvus collections/indexes, Kafka topics, and Graph/LLM endpoints checks).
5. **Phase 27: Graceful Shutdown** (Graceful termination of consumer loops, offset commits, connection pools shutdown).

---

## 1. Executive Summary

As the Memory Service progresses toward production readiness, we have completed the integration of clients, security controls, startup validation routines, and graceful shutdown lifecycles.

Key Highlights of this update:
* **Phase 23 & 24 Client Resilience**: The Graph Client now uses connection pooling and retries with backoff, protected by a circuit breaker. Embedding generation has been decoupled into pluggable adapters (Mock/gRPC) to enable offline testing and smooth future model replacements.
* **Phase 25 Service-to-Service Auth**: Added robust, dependency-free HS256 JWT decoding and API Key validation. Secured the core `/internal/memory/context` endpoint.
* **Phase 26 Startup checks**: Added a deep validation sequence that executes on app startup, verifying keyspace tables, Redis connectivity, Milvus vectors, Kafka topics, and external service reachability.
* **Phase 27 Clean Shutdowns**: Redesigned all background worker and consumer loops (`SummaryWorker`, `FactWorker`, `EmbeddingWorker`, `DeleteWorker`, and `KafkaEventConsumer`) to stop accepting new items, finish any active in-flight jobs, and commit Kafka partition offsets cleanly before the process exits.

All 121 unit tests pass.

---

## 2. Phase 23: Graph Service Client

To make downstream HTTP requests resilient to transient failures and avoid socket exhaustion, we refactored the `GraphClient` in `app/clients/graph_client.py`.

### 2.1 Connection Pooling and Shared Session
Instead of spawning a new `httpx.AsyncClient` on every request, the `GraphClient` creates a single shared `httpx.AsyncClient` session during `connect()` and re-uses it across all requests, utilizing internal HTTP connection pooling.

### 2.2 Exponential Backoff and Retries
HTTP requests to the Graph Service are wrapped in a retry handler that targets transient errors (such as network drops, timeout exceptions, and HTTP 5xx responses).
* **Strategy**: Retries are attempted up to 3 times.
* **Backoff**: An exponential backoff calculation (`2^attempt + jitter`) is used to space out retries and prevent overwhelming the destination service.

### 2.3 HTTP Client Circuit Breaker
The `GraphClient` includes a dedicated consecutive-failure state-based circuit breaker matching the requirements in `docs/lld.md`:
* **State transition rules**:
  - `CLOSED`: Normal operations. If consecutive failures exceed `settings.CB_FAILURE_THRESHOLD` (5), transitions to `OPEN`.
  - `OPEN`: Fast-fails requests, raising `CircuitBreakerOpenException`. Stays open for `settings.CB_RECOVERY_TIMEOUT_SECONDS` (60s).
  - `HALF_OPEN`: Probes the service with up to `settings.CB_HALF_OPEN_LIMIT` (2) requests. If any fails, transitions back to `OPEN`. If all succeed, transitions to `CLOSED` and resets failure counts.

---

## 3. Phase 24: Embedding Client Abstraction

To decouple vector embedding operations from the general LLM gRPC pool, we introduced a pluggable client adapter architecture.

### 3.1 Interface Definition
The abstract class `EmbeddingClient` defines the contract for all embedding clients:
```python
class EmbeddingClient(ABC):
    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass
```

### 3.2 Implemented Adapters
1. **`GRPCEmbeddingClient`**:
   - Communicates with the external LLM Service using gRPC stubs.
   - Manages a pool of persistent async gRPC channels.
   - Guarded by a dedicated circuit breaker.
2. **`MockEmbeddingClient`**:
   - Returns mock vectors (e.g. cosine-normalized mock arrays) of size `settings.VECTOR_DIMENSION` (1536).
   - Designed for offline development and testing modes to avoid network/external dependencies.

### 3.3 Configuration and DI Container Integration
* A setting `EMBEDDING_CLIENT_TYPE: str` ("grpc" or "mock") was introduced in config.
* The DI container (`Container`) reads this setting and instantiates the correct client adapter during `init_resources()`.
* `EmbeddingWorker` was refactored to consume the abstract `EmbeddingClient` dependency, completely insulating business logic from the communication protocol.

---

## 4. Phase 25: Service-to-Service Security

The Memory Service context API (`POST /internal/memory/context`) serves raw cognitive memories to the LLM. Securing it from unauthorized internal callers is critical.

### 4.1 Dependency-Free Cryptographic JWT Helpers
To keep the service lightweight and avoid importing third-party JWT libraries, we implemented standard HS256 JWT signature verification from scratch in `app/core/security.py` using standard python library functions (`hmac`, `hashlib`, `base64`):

```python
def verify_jwt(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header_segment, payload_segment, signature_segment = parts
        
        # Verify signature integrity
        signing_input = (header_segment + "." + payload_segment).encode("utf-8")
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            signing_input,
            hashlib.sha256
        ).digest()
        actual_sig = base64url_decode(signature_segment)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
            
        # Decode and verify expiration
        payload = json.loads(base64url_decode(payload_segment).decode("utf-8"))
        exp = payload.get("exp")
        if exp and exp < time.time():
            return None
            
        return payload
    except Exception:
        return None
```

### 4.2 FastAPI Dependency Security
We added `verify_service_auth` as a security dependency:
* **API Key Auth**: Checks if the header `X-API-Key` matches `settings.API_KEY`.
* **JWT Bearer Auth**: Checks if the `Authorization` header has a valid Bearer token and verifies the signature using the service's `JWT_SECRET_KEY`.
* Unauthenticated requests raise `HTTP 401 Unauthorized`.

---

## 5. Phase 26: Startup Validation

To ensure the application never enters a running state with missing or offline resources, we created a comprehensive startup validation pipeline in `app/core/startup_validation.py`.

### 5.1 Validation Routines
1. **Cassandra Schema Validation**: Resolves metadata columns and tables (via `MigrationManager`). Verifies `snapshot_metadata` and the new `created_at` in the outbox index exist.
2. **Redis Connectivity**: Pings the Redis cache to ensure sub-millisecond keys operations are available.
3. **Milvus Vectors Check**: Confirms `user_memory_vectors` is present and HNSW index has been successfully loaded into memory.
4. **Kafka Topics Check**: Fetches topics list from the brokers and verifies that expected consumer/producer topics exist.
5. **Downstream Reachability**: Checks HTTP endpoint reachability for the Graph Service and gRPC connectivity for the LLM Service.
   - **Non-Strict Mode (Default)**: Logs warnings if Graph/LLM are offline, preventing blocked local development when downstream systems are down.
   - **Strict Mode**: If `settings.STRICT_STARTUP_VALIDATION = True`, downstream failures raise a `RuntimeError` and halt startup.

---

## 6. Phase 27: Graceful Shutdown

Immediate task cancellations during shutdown leave worker states corrupted and duplicate Kafka messages because partition offsets aren't committed.

### 6.1 Implementation
We updated the shutdown logic inside the `stop()` method of the `KafkaEventConsumer` and background workers (`SummaryWorker`, `FactWorker`, `EmbeddingWorker`, `DeleteWorker`):
* `self.is_running = False` is set, telling the loops to stop polling.
* A timeout-based wait using `asyncio.wait_for` and `asyncio.shield` is implemented. It waits up to 10 seconds for the active message iteration to complete.
* If a message is being processed, the worker completes it, saves database changes to Cassandra, commits offsets to Kafka, and exits the loop cleanly.
* If the loop remains hung after 10 seconds, it falls back to cancelling the task.
* Graceful closing of all connection pools (gRPC channels, Redis pools, Cassandra session, Milvus connections) occurs in reverse initialization order.

```python
async def stop(self) -> None:
    logger.info("Initiating graceful shutdown...")
    self.is_running = False
    if self._task:
        try:
            # Wait up to 10 seconds for active loop to complete
            await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            logger.info("Worker loop exited cleanly.")
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timed out. Forcefully cancelling loop.")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass
        self._task = None
```

---

## 7. Verification and Test Results

We expanded our unit tests by 15 test cases to cover security, startup verification, and graceful shutdown:
1. **`tests/unit/test_security.py`**:
   - `test_base64url_helpers`: Verified URL-safe base64 encoding/decoding.
   - `test_generate_and_verify_jwt_success`: Validated signature matching.
   - `test_verify_jwt_invalid_signature`: Verified tampered token rejection.
   - `test_verify_jwt_expired`: Verified expired token rejection.
   - `test_verify_service_auth_api_key`: Checked header API Key verification.
   - `test_verify_service_auth_jwt`: Checked Bearer JWT token verification.
   - `test_verify_service_auth_unauthorized`: Checked 401 raises.
2. **`tests/unit/test_startup_validation.py`**:
   - Verified happy-paths, critical database/broker failures, and strict/non-strict behavior variations.
3. **`tests/unit/test_background_workers.py`**:
   - `test_summary_worker_graceful_shutdown`: Verified that `stop()` allows in-flight summaries to complete and commit partition offsets before stopping.
4. **`tests/unit/test_api_endpoints.py`**:
   - Verified that unauthenticated requests to `/internal/memory/context` fail with `401 Unauthorized`, while correct keys/tokens return `200 OK`.

All 121 unit tests passed successfully.
