"""
File: preflight.py

Description:
Convenience wrapper for the application health check.

Responsibilities:
- Run environment, configuration, disk, and Ollama checks without starting a benchmark.
"""

from benchmark.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["health"]))
