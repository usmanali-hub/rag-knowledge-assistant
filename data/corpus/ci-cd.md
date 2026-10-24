# Continuous integration and delivery

Continuous integration runs the test suite automatically on every push, so a broken change is caught within minutes instead of weeks. A workflow file in the repository defines the steps: check out the code, install dependencies, lint, type-check and run the tests. Caching dependencies speeds up runs. Continuous delivery goes further and builds a release artifact, such as a container image, and deploys it when the checks pass. Keep the pipeline fast and deterministic, because a flaky test that people learn to ignore is worse than no test.
