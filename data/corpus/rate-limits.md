# Rate limits and retries

Model APIs limit how many requests and tokens a client may send per minute. When the limit is hit the server returns HTTP status 429, often with a header saying how long to wait. Clients should retry with exponential backoff, doubling the delay after each failure and adding random jitter so many clients do not retry at the same instant. Retries need a cap on attempts, and only idempotent or safely repeatable calls should be retried. A queue in front of the API smooths bursts, and respecting the retry header is better than guessing.
