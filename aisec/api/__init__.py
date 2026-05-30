"""
AISec REST API.

Enterprise HTTP interface for the AISec analysis engine.
Any language, any platform, any AI framework can integrate
with AISec over HTTP without the Python SDK.

Endpoints:
    POST /api/v1/analyse          — analyse a single event
    POST /api/v1/analyse/batch    — analyse up to 100 events
    GET  /api/v1/health           — liveness and readiness
    GET  /api/v1/audit/verify     — hash chain integrity
    GET  /api/v1/metrics/summary  — security metrics

Start the server:
    aisec serve
    aisec serve --host 0.0.0.0 --port 8000

API documentation:
    http://localhost:8000/docs       Swagger UI
    http://localhost:8000/redoc      ReDoc
    http://localhost:8000/openapi.json
"""
