# DCI with vLLM on the university GPU

## 1. Check GPU

```bash
nvidia-smi
```

Expected: RTX 4090 · 24,564 MiB · driver 595.99.02 · CUDA 13.2.

## 2. First setup

```bash
cd ~/muazzam
git clone --recurse-submodules https://github.com/Master-ISY-Coursework/Semester-2-Seminar.git
cd Semester-2-Seminar/external/DCI-Agent-Lite

curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
source "$HOME/.local/bin/env"

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install 20.20.2

uv sync
```

Expected: uv 0.13.7 · Node v20.20.2 · npm 10.8.2.

## 3. Build Pi once

Skip the clone command if `pi-mono/package.json` already exists.

```bash
git clone https://github.com/jdf-prog/pi-mono.git pi-mono
cd pi-mono
git checkout codex/context-management-ablation
npm install
npm run build
cd ..

node pi-mono/packages/coding-agent/dist/cli.js --version
```

Expected: `0.65.0`.

## 4. Download and prepare data once

```bash
uv run python scripts/download_corpus.py
uv run python scripts/download_dci_bench.py
uv run python scripts/bcplus_eval/extract_bcplus_qa.py

ls -lh corpus/wiki_corpus/wiki_dump.jsonl
wc -l data/bcplus_qa.jsonl
```

Expected:

- BrowseComp-Plus: 100,195 exported documents in `corpus/bc_plus_docs/`.
- Wikipedia: `corpus/wiki_corpus/wiki_dump.jsonl` · approximately 14 GB.
- Questions: 830 records in `data/bcplus_qa.jsonl`.

## 5. Install vLLM once

```bash
uv venv "$HOME/.venvs/vllm" --python 3.12
uv pip install --python "$HOME/.venvs/vllm/bin/python" 'vllm==0.33.0' --torch-backend=auto
source "$HOME/.venvs/vllm/bin/activate"
vllm download-kernels
```

Expected: vLLM 0.33.0 and matching FlashInfer kernels installed.

## 6. Configure Pi once

```bash
mkdir -p "$HOME/.pi/agent"
cat > "$HOME/.pi/agent/models.json" <<'JSON'
{
  "providers": {
    "vllm": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "VLLM_API_KEY",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [{
        "id": "Qwen/Qwen3-1.7B",
        "reasoning": true,
        "contextWindow": 16384,
        "maxTokens": 8192
      }]
    }
  }
}
JSON

cat > "$HOME/.pi/agent/settings.json" <<'JSON'
{
  "lastChangelogVersion": "0.65.0",
  "compaction": {
    "enabled": true,
    "reserveTokens": 8192,
    "keepRecentTokens": 2048
  }
}
JSON
```

## 7. Every session — Terminal 1

```bash
source "$HOME/.venvs/vllm/bin/activate"

vllm serve Qwen/Qwen3-1.7B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

Expected: `Application startup complete.` Keep this terminal running.

## 8. Every session — Terminal 2

```bash
cd ~/muazzam/Semester-2-Seminar/external/DCI-Agent-Lite
source "$HOME/.local/bin/env"
source "$HOME/.nvm/nvm.sh"
nvm use 20.20.2
export VLLM_API_KEY=dummy

curl --fail http://127.0.0.1:8000/v1/models
```

Expected: JSON containing `Qwen/Qwen3-1.7B` and `"max_model_len":16384`.

## 9. Check tool execution — Terminal 2

```bash
uv run dci-agent-lite \
  --provider vllm \
  --model Qwen/Qwen3-1.7B \
  --agent-dir "$HOME/.pi/agent" \
  --cwd "$PWD/corpus/bc_plus_docs" \
  --extra-arg="--context-management-level level3" \
  "Use bash to run pwd and return the result. /no_think"
```

Expected: `/home/jovyan/work/muazzam/Semester-2-Seminar/external/DCI-Agent-Lite/corpus/bc_plus_docs`.

## 10. Test 1 — BrowseComp-Plus

```bash
QUESTION=$(python3 -c 'import json; print(json.loads(open("data/bcplus_qa.jsonl").readline())["query"])')

uv run dci-agent-lite \
  --terminal \
  --provider vllm \
  --model Qwen/Qwen3-1.7B \
  --agent-dir "$HOME/.pi/agent" \
  --cwd "$PWD/corpus/bc_plus_docs" \
  --extra-arg="--context-management-level level3" \
  "Answer using only the local corpus. Use rg for searching. Do not use web search. Question: $QUESTION"
```

- Reference: Queen Arwa University · query ID `769`.
- Observed: invented filenames and an incorrect answer of University of California, Berkeley.
- Earlier `/no_think` trial: invented `learning_institution.txt` and stopped.

Exit Pi with **Ctrl+D** on empty input before the next test.

## 11. Test 2 — Wikipedia

```bash
uv run dci-agent-lite \
  --terminal \
  --provider vllm \
  --model Qwen/Qwen3-1.7B \
  --agent-dir "$HOME/.pi/agent" \
  --cwd "$PWD/corpus/wiki_corpus" \
  --extra-arg="--context-management-level level3" \
  "Answer the following question using only wiki_dump.jsonl in the current directory. Do not use web search. Use rg instead of grep for fast searching. Question: In which street did the Great Fire of London originate?"
```

- Reference: Pudding Lane.
- Observed: repeated `read` calls failed with `File size (14393573105) is greater than 2 GiB`.

Press **Esc** to interrupt. Paste into Pi:

```text
Do not use read on wiki_dump.jsonl. Use bash to execute:
rg -n -i -m 5 'Great Fire of London' wiki_dump.jsonl | cut -c 1-1500

Use the search results to answer the question.
```

Observed: search succeeded; the model incorrectly answered Steelyard.

Paste into Pi:

```text
Run this using bash:

rg -i 'Great Fire of London' wiki_dump.jsonl | rg -i 'began|started|originated|bakery' | head -n 5 | cut -c 1-1500

Answer only if a passage explicitly identifies the origin street. Include its document title and record ID.
```

- Expected evidence: document `Billingsgate` · record ID `1250555` · Pudding Lane is explicitly identified.
- Observed: search succeeded; the model incorrectly answered Billingsgate.
- Keep these guided attempts separate from the original test.

## 12. Pi controls

- **Ctrl+O:** expand tool output.
- **Ctrl+T:** expand thinking.
- **Esc:** interrupt generation.
- **Ctrl+D** on empty input: exit Pi.
- `--show-tools`: start/end events in programmatic mode; omit with `--terminal`.
