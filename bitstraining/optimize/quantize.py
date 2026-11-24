"""GGUF export through llama.cpp's own tools (this module only builds and runs the commands).

Requires a llama.cpp checkout (set LLAMA_CPP_DIR) with `convert_hf_to_gguf.py` and a built `llama-quantize`.
NOT exercised by CI; `gguf_command` is a pure function and is unit-tested.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from bitstraining.core.errors import EngineUnavailable

QUANTS = {"f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m", "q2_k"}


def gguf_command(model_dir: str, out_file: str, quant: str, llama_cpp: str) -> list[list[str]]:
    """The commands (convert to f16 GGUF, then quantize) for a merged model directory."""
    if quant not in QUANTS:
        raise ValueError(f"unknown quant '{quant}'; choose from {sorted(QUANTS)}")
    f16 = str(Path(out_file).with_suffix("")) + ".f16.gguf"
    convert = [sys.executable, str(Path(llama_cpp) / "convert_hf_to_gguf.py"), model_dir, "--outfile",
               f16, "--outtype", "f16"]
    if quant == "f16":
        return [convert[:-2] + ["--outfile", out_file, "--outtype", "f16"]]
    return [convert, [str(Path(llama_cpp) / "build" / "bin" / "llama-quantize"), f16, out_file, quant.upper()]]


def quantize_gguf(model_dir: str, out_file: str, quant: str = "q4_k_m") -> Path:
    root = os.environ.get("LLAMA_CPP_DIR")
    if not root or not (Path(root) / "convert_hf_to_gguf.py").exists():
        raise EngineUnavailable("llama.cpp not found", fix="clone llama.cpp, build it, and set LLAMA_CPP_DIR")
    for cmd in gguf_command(model_dir, out_file, quant, root):
        subprocess.run(cmd, check=True)
    return Path(out_file)
