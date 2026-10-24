# Python type hints

Type hints document what a function expects and returns, and a checker such as mypy finds mismatches before the code runs. They have no effect at runtime, so a wrong hint will not raise an error by itself. Use built-in generics like list of str and dict of str to int, and a union with None for optional values. Protocols describe behaviour instead of inheritance, so any object with the right methods fits. Strict mode forbids untyped definitions and implicit Any, which keeps a codebase from sliding back into guesswork.
