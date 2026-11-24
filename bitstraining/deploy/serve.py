"""Build the command that serves a model with an OpenAI-compatible API (vLLM or llama.cpp).

Bitstraining does not reimplement a server; it assembles and (optionally) runs the right one.
"""

from __future__ import annotations


def serve_command(model: str, backend: str = "vllm", port: int = 8000, host: str = "127.0.0.1",
                  max_len: int | None = None) -> list[str]:
    if backend == "vllm":
        cmd = ["vllm", "serve", model, "--host", host, "--port", str(port)]
        if max_len:
            cmd += ["--max-model-len", str(max_len)]
        return cmd
    if backend == "llamacpp":
        cmd = ["llama-server", "-m", model, "--host", host, "--port", str(port)]
        if max_len:
            cmd += ["-c", str(max_len)]
        return cmd
    raise ValueError(f"unknown backend '{backend}' (use vllm or llamacpp)")
