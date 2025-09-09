# Qdrant Vector Database API Specification

## Overview
High-performance vector database service for storing and retrieving 512-dimensional vector embeddings with optional GPU acceleration.

## Base URL
```
http://localhost:8001/api/v1
```

## Endpoints

### 1) Add Vector
- Method: POST
- Path: `/vectors/add`
- Request Body:
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "user_id": "user123",
  "metadata": {"key": "value"},
  "point_id": "optional-custom-id"
}
```
- Response (200):
```json
{
  "success": true,
  "message": "Vector added successfully",
  "timestamp": 1725900000.0,
  "point_id": "generated-or-provided-id"
}
```

### 2) Add Vectors (Batch)
- Method: POST
- Path: `/vectors/add_batch`
- Request Body:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "user_ids": ["user1", "user2"],
  "metadata_list": [{"group": "A"}, {"group": "B"}],
  "point_ids": ["id1", "id2"]
}
```
- Response (200):
```json
{
  "success": true,
  "message": "Successfully added 2 vectors",
  "timestamp": 1725900000.0,
  "point_ids": ["id1", "id2"],
  "added_count": 2
}
```

### 3) Search Vectors
- Method: POST
- Path: `/vectors/search`
- Request Body:
```json
{
  "embedding": [0.1, 0.2, ...],
  "k": 10,
  "threshold": 0.65,
  "user_filter": "user123"
}
```
- Response (200):
```json
{
  "success": true,
  "message": "Found 3 similar vectors",
  "timestamp": 1725900000.0,
  "results": [
    {
      "id": "point-id",
      "score": 0.95,
      "user_id": "user123",
      "metadata": {"key": "value"},
      "timestamp": 1725900000.0
    }
  ],
  "query_time_ms": 12.34,
  "total_results": 3
}
```

### 4) Delete Vector
- Method: DELETE
- Path: `/vectors/{point_id}`
- Response (200):
```json
{
  "success": true,
  "message": "Vector <point_id> deleted",
  "timestamp": 1725900000.0,
  "deleted": true
}
```

### 5) Delete All Vectors for a User
- Method: DELETE
- Path: `/vectors/user/{user_id}`
- Response (200):
```json
{
  "success": true,
  "message": "Deleted N vectors for user <user_id>",
  "timestamp": 1725900000.0,
  "deleted_count": 42
}
```

### 6) Health Check
- Method: GET
- Path: `/health`
- Response (200):
```json
{
  "success": true,
  "status": "healthy",
  "qdrant_connection": true,
  "collection_exists": true,
  "collection_name": "face_embeddings",
  "gpu_available": true,
  "message": "Service is healthy",
  "timestamp": 1725900000.0
}
```

### 7) Statistics
- Method: GET
- Path: `/stats`
- Response (200):
```json
{
  "success": true,
  "message": "Statistics retrieved successfully",
  "timestamp": 1725900000.0,
  "collection_info": {"vector_count": 1000},
  "performance_stats": {"avg_search_time_ms": 15.2},
  "gpu_info": {"available": true}
}
```

### 8) Service Info
- Method: GET
- Path: `/info`
- Response (200):
```json
{
  "service_name": "Qdrant Vector Database Service",
  "version": "1.0.0",
  "description": "High-performance GPU-accelerated vector database service",
  "configuration": {
    "vector_dimension": 512,
    "similarity_metric": "Cosine",
    "similarity_threshold": 0.65,
    "collection_name": "face_embeddings",
    "gpu_enabled": true,
    "cuda_available": true,
    "batch_size": 128
  },
  "endpoints": {
    "vectors": [
      "/vectors/add",
      "/vectors/add_batch",
      "/vectors/search",
      "/vectors/{point_id}",
      "/vectors/user/{user_id}"
    ],
    "system": [
      "/health",
      "/stats",
      "/info"
    ]
  },
  "timestamp": 1725900000.0
}
```

## Schemas (Summarized)

### AddVectorRequest
- `embedding`: List[float] (512-dim)
- `user_id`: str
- `metadata`: Dict[str, Any] (optional)
- `point_id`: str (optional)

### AddVectorResponse
- Inherits `BaseResponse`
- `point_id`: str

### AddVectorsBatchRequest
- `embeddings`: List[List[float]]
- `user_ids`: List[str]
- `metadata_list`: List[Dict[str, Any]] (optional)
- `point_ids`: List[str] (optional)

### AddVectorsBatchResponse
- Inherits `BaseResponse`
- `point_ids`: List[str]
- `added_count`: int

### SearchRequest
- `embedding`: List[float]
- `k`: int (1-100)
- `threshold`: float (0.0-1.0, optional)
- `user_filter`: str (optional)

### SearchResponse
- Inherits `BaseResponse`
- `results`: List[SearchResult]
- `query_time_ms`: float
- `total_results`: int

### SearchResult
- `id`: str
- `score`: float
- `user_id`: str
- `metadata`: Dict[str, Any]
- `timestamp`: float (optional)

### DeleteVectorResponse
- Inherits `BaseResponse`
- `deleted`: bool

### DeleteUserVectorsResponse
- Inherits `BaseResponse`
- `deleted_count`: int

### StatsResponse
- Inherits `BaseResponse`
- `collection_info`: Dict[str, Any]
- `performance_stats`: Dict[str, Any]
- `gpu_info`: Dict[str, Any]

### HealthCheckResponse
- Inherits `BaseResponse`
- `status`: str
- `qdrant_connection`: bool
- `collection_exists`: bool
- `collection_name`: str
- `gpu_available`: bool

## Notes
- API prefix is `/api/v1` (see `src/main.py`).
- Default `collection_name` is `face_embeddings` (see `src/config/settings.py`).
- Default vector dimension is 512 (`settings.vector_size`).
- Responses include `timestamp` where set by endpoints via `BaseResponse`.

Below is a concise, ready-to-use API specification you can import into tools like Postman/Insomnia or generate clients from. It reflects the implemented FastAPI endpoints in your codebase:

- Base app: [src/main.py](cci:7://file:///d:/RnD/face-rnd/qrant-service/src/main.py:0:0-0:0)
- Router: [src/api/endpoints.py](cci:7://file:///d:/RnD/face-rnd/qrant-service/src/api/endpoints.py:0:0-0:0) under prefix `/api/v1`
- Schemas: [src/api/schemas.py](cci:7://file:///d:/RnD/face-rnd/qrant-service/src/api/schemas.py:0:0-0:0)

You’ll find:
- Overview and base URL
- Endpoint summaries with request/response examples
- A complete OpenAPI 3.0 YAML you can save as openapi.yaml

API Overview

- Base URL: http://localhost:8001
- API Prefix: /api/v1
- Auth: None (you may add later)
- Content-Type: application/json
- Vector dimension: 512 (float32)
- Similarity: Cosine similarity; default threshold 0.35

Endpoints Summary

- Vectors
  - POST /api/v1/vectors/add — add one vector
  - POST /api/v1/vectors/add_batch — add vectors in batch
  - POST /api/v1/vectors/search — search similar vectors
  - DELETE /api/v1/vectors/{point_id} — delete by point id
  - DELETE /api/v1/vectors/user/{user_id} — delete all of a user’s vectors

- Face Recognition (compatibility)
  - POST /api/v1/enroll — enroll a user with embedding
  - POST /api/v1/verify — verify a face against a specific user
  - POST /api/v1/detect — identify top-k candidates

- System
  - GET /api/v1/health — service health
  - GET /api/v1/stats — collection stats + GPU info
  - GET /api/v1/info — service info
  - GET /benchmark — run a lightweight benchmark (no prefix)

OpenAPI 3.0 Specification (save as openapi.yaml)

```yaml
openapi: 3.0.3
info:
  title: Qdrant Vector Database Service
  version: "1.0.0"
  description: >
    GPU-accelerated vector database service for face recognition.
    Prefix: /api/v1
servers:
  - url: http://localhost:8001
    description: Local server
tags:
  - name: Vectors
  - name: Face Recognition
  - name: System

paths:
  /api/v1/vectors/add:
    post:
      tags: [Vectors]
      summary: Add a single embedding vector
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/AddVectorRequest' }
            examples:
              default:
                value:
                  embedding: [0.01, 0.02, 0.03, ...]  # 512 floats
                  user_id: user_123
                  metadata: { name: "John Doe" }
      responses:
        "200":
          description: Vector added
          content:
            application/json:
              schema: { $ref: '#/components/schemas/AddVectorResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/vectors/add_batch:
    post:
      tags: [Vectors]
      summary: Add multiple vectors in batch
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/AddVectorsBatchRequest' }
            examples:
              default:
                value:
                  embeddings: [[0.01, 0.02, ...], [0.11, 0.12, ...]]
                  user_ids: ["user1", "user2"]
                  metadata_list: [{ role: "admin" }, { role: "staff" }]
      responses:
        "200":
          description: Batch add result
          content:
            application/json:
              schema: { $ref: '#/components/schemas/AddVectorsBatchResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/vectors/search:
    post:
      tags: [Vectors]
      summary: Search similar vectors
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/SearchRequest' }
            examples:
              default:
                value:
                  embedding: [0.01, 0.02, ...]
                  k: 5
                  threshold: 0.35
                  user_filter: null
      responses:
        "200":
          description: Search results
          content:
            application/json:
              schema: { $ref: '#/components/schemas/SearchResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/vectors/{point_id}:
    delete:
      tags: [Vectors]
      summary: Delete a vector by point ID
      parameters:
        - in: path
          name: point_id
          required: true
          schema: { type: string }
      responses:
        "200":
          description: Deletion result
          content:
            application/json:
              schema: { $ref: '#/components/schemas/DeleteVectorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/vectors/user/{user_id}:
    delete:
      tags: [Vectors]
      summary: Delete all vectors for a user
      parameters:
        - in: path
          name: user_id
          required: true
          schema: { type: string }
      responses:
        "200":
          description: Deletion count
          content:
            application/json:
              schema: { $ref: '#/components/schemas/DeleteUserVectorsResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/enroll:
    post:
      tags: [Face Recognition]
      summary: Enroll a user with a face embedding
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/EnrollRequest' }
      responses:
        "200":
          description: Enrollment success
          content:
            application/json:
              schema: { $ref: '#/components/schemas/EnrollResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/verify:
    post:
      tags: [Face Recognition]
      summary: Verify a face against a specific user
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/VerifyRequest' }
      responses:
        "200":
          description: Verification result
          content:
            application/json:
              schema: { $ref: '#/components/schemas/VerifyResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/detect:
    post:
      tags: [Face Recognition]
      summary: Identify top-k candidates for a face embedding
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/DetectRequest' }
      responses:
        "200":
          description: Detection candidates
          content:
            application/json:
              schema: { $ref: '#/components/schemas/DetectResponse' }
        "400":
          description: Validation error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }
        "500":
          description: Server error
          content:
            application/json:
              schema: { $ref: '#/components/schemas/ErrorResponse' }

  /api/v1/health:
    get:
      tags: [System]
      summary: Health check
      responses:
        "200":
          description: Health status
          content:
            application/json:
              schema: { $ref: '#/components/schemas/HealthCheckResponse' }

  /api/v1/stats:
    get:
      tags: [System]
      summary: Collection and performance stats
      responses:
        "200":
          description: Stats payload
          content:
            application/json:
              schema: { $ref: '#/components/schemas/StatsResponse' }

  /api/v1/info:
    get:
      tags: [System]
      summary: Service info
      responses:
        "200":
          description: Info payload
          content:
            application/json:
              schema:
                type: object
                additionalProperties: true

  /benchmark:
    get:
      tags: [System]
      summary: Run a lightweight benchmark (dev/testing)
      responses:
        "200":
          description: Benchmark results
          content:
            application/json:
              schema:
                type: object
                properties:
                  success: { type: boolean }
                  benchmark_results: { type: object }
                  timestamp: { type: number }

components:
  schemas:
    BaseResponse:
      type: object
      properties:
        success: { type: boolean }
        message: { type: string }
        timestamp: { type: number, nullable: true }

    ErrorResponse:
      type: object
      properties:
        success: { type: boolean, default: false }
        error: { type: string }
        error_code: { type: string, nullable: true }
        details: { type: object, nullable: true, additionalProperties: true }
        timestamp: { type: number }

    AddVectorRequest:
      type: object
      required: [embedding, user_id]
      properties:
        embedding:
          type: array
          items: { type: number, format: float }
          minItems: 512
          maxItems: 512
        user_id: { type: string }
        metadata:
          type: object
          additionalProperties: true
        point_id: { type: string, nullable: true }

    AddVectorResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            point_id: { type: string, nullable: true }

    AddVectorsBatchRequest:
      type: object
      required: [embeddings, user_ids]
      properties:
        embeddings:
          type: array
          items:
            type: array
            items: { type: number, format: float }
            minItems: 512
            maxItems: 512
        user_ids:
          type: array
          items: { type: string }
        metadata_list:
          type: array
          items:
            type: object
            additionalProperties: true
          nullable: true
        point_ids:
          type: array
          items: { type: string }
          nullable: true

    AddVectorsBatchResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            point_ids:
              type: array
              items: { type: string }
            added_count: { type: integer }

    SearchRequest:
      type: object
      required: [embedding]
      properties:
        embedding:
          type: array
          items: { type: number, format: float }
          minItems: 512
          maxItems: 512
        k:
          type: integer
          minimum: 1
          maximum: 100
          default: 10
        threshold:
          type: number
          format: float
          minimum: 0
          maximum: 1
          nullable: true
        user_filter:
          type: string
          nullable: true

    SearchResult:
      type: object
      properties:
        id: { type: string }
        score: { type: number, format: float }
        user_id: { type: string }
        metadata:
          type: object
          additionalProperties: true
        timestamp: { type: number, nullable: true }

    SearchResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            results:
              type: array
              items: { $ref: '#/components/schemas/SearchResult' }
            query_time_ms: { type: number, format: float }
            total_results: { type: integer }

    DeleteVectorResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            deleted: { type: boolean }

    DeleteUserVectorsResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            deleted_count: { type: integer }

    EnrollRequest:
      type: object
      required: [user_id, embedding]
      properties:
        user_id: { type: string }
        embedding:
          type: array
          items: { type: number, format: float }
          minItems: 512
          maxItems: 512
        metadata:
          type: object
          nullable: true
          additionalProperties: true

    EnrollResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            user_id: { type: string }
            enrollment_id: { type: string }

    VerifyRequest:
      type: object
      required: [user_id, embedding]
      properties:
        user_id: { type: string }
        embedding:
          type: array
          items: { type: number, format: float }
          minItems: 512
          maxItems: 512
        threshold:
          type: number
          format: float
          nullable: true

    VerifyResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            verified: { type: boolean }
            confidence: { type: number, format: float }
            user_id: { type: string }
            threshold_used: { type: number, format: float }

    DetectRequest:
      type: object
      required: [embedding]
      properties:
        embedding:
          type: array
          items: { type: number, format: float }
          minItems: 512
          maxItems: 512
        k:
          type: integer
          minimum: 1
          maximum: 50
          default: 5
        threshold:
          type: number
          format: float
          nullable: true

    DetectionCandidate:
      type: object
      properties:
        user_id: { type: string }
        confidence: { type: number, format: float }
        metadata:
          type: object
          additionalProperties: true

    DetectResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            candidates:
              type: array
              items: { $ref: '#/components/schemas/DetectionCandidate' }
            query_time_ms: { type: number, format: float }
            total_candidates: { type: integer }

    StatsResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            collection_info:
              type: object
              additionalProperties: true
            performance_stats:
              type: object
              additionalProperties: true
            gpu_info:
              type: object
              additionalProperties: true

    HealthCheckResponse:
      allOf:
        - $ref: '#/components/schemas/BaseResponse'
        - type: object
          properties:
            status: { type: string, enum: [healthy, unhealthy] }
            qdrant_connection: { type: boolean }
            collection_exists: { type: boolean }
            collection_name: { type: string }
            gpu_available: { type: boolean }
```

Usage Notes

- The FastAPI app already exposes interactive docs at /docs and /redoc.
- The benchmark endpoint is intentionally outside the /api/v1 prefix as implemented in [src/main.py](cci:7://file:///d:/RnD/face-rnd/qrant-service/src/main.py:0:0-0:0).

If you’d like, I can:
- Export this spec into your repo as openapi.yaml.
- Generate typed API clients (e.g., TypeScript, Python).
- Add JWT auth/security schemes to the spec if you plan to secure these endpoints.