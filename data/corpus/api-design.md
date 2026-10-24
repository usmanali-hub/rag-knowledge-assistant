# Designing a model-serving API

A model service should expose a small, versioned HTTP API with a clear request and response schema. Validate input with a schema library so malformed requests fail fast with a helpful message instead of reaching the model. Add a health endpoint for load balancers and set a request timeout, because model calls can hang. Stream long replies token by token so users see output early. Return stable error codes, include a request identifier in every response for tracing, and keep secrets such as API keys in environment variables rather than in the code.
