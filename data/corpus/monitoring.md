# Monitoring deployed AI services

A deployed model needs the same monitoring as any service, plus checks specific to AI. Track request rate, error rate and latency percentiles such as p95, since averages hide slow tails. Log token usage and cost per request. Quality drift is harder to see, so sample real traffic, store the question, retrieved context and answer, and review them regularly or score them with an automated judge. Alert on sudden changes in refusal rate, empty retrievals and answer length. Dashboards without alerts only help after the incident has already happened.
