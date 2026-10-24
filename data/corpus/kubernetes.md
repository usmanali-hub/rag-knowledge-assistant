# Kubernetes basics

Kubernetes runs containers across a cluster of machines and keeps them in the desired state. A pod is the smallest unit and holds one or more containers. A deployment declares how many identical pods should run and replaces them during a rolling update, so a new version can be released without downtime. A service gives pods a stable network address and balances traffic between them. Resource requests and limits tell the scheduler how much CPU and memory a pod needs, and readiness probes stop traffic reaching a pod that is not yet able to answer.
