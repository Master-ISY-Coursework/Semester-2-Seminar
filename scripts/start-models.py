#!/usr/bin/env python3
"""Serve configured models through stable Ollama API names."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

CONFIG_PATH = Path(__file__).resolve().parents[1] / "settings/llm/models.json"
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if "://" not in HOST:
    HOST = "http://" + HOST


def api(path, payload=None):
    """Send a request to Ollama and return its JSON response."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        HOST + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    models = config["models"]

    # Show NVIDIA hardware. Ollama chooses GPU or CPU automatically.
    if shutil.which("nvidia-smi"):
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False, timeout=15,
        )
    else:
        print("No NVIDIA tool detected; Ollama will select available GPU/CPU hardware.")
    installed = {m["name"] for m in api("/api/tags")["models"]}
    for alias, source in models.items():
        if source not in installed:
            print(f"Downloading {source}", flush=True)
            api("/api/pull", {"model": source, "stream": False})

        # Keep the API name fixed when the source model changes.
        api("/api/create", {
            "model": alias, "from": source, "stream": False,
            "parameters": {"num_ctx": config["context_tokens"]},
        })
        # Load the model without a prompt and disable its idle timeout.
        api("/api/generate", {"model": alias, "keep_alive": -1, "stream": False})

    # Confirm both models are loaded and report GPU memory placement.
    running = {m["name"]: m for m in api("/api/ps")["models"]}
    for alias in models:
        model_name = alias if ":" in alias else f"{alias}:latest"
        model = running.get(model_name)
        if not model:
            raise RuntimeError(f"{alias} is not loaded; both models may not fit in memory.")
        gpu_percent = round(100 * model["size_vram"] / model["size"])
        print(f"{alias} ready: {gpu_percent}% GPU / {100 - gpu_percent}% CPU")

    print(f"OpenAI-compatible API: {HOST}/v1")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
