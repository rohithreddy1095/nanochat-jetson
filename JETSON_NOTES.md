# NanoChat on Jetson Orin — Setup & Reference

## System Specs
- **Device**: NVIDIA Jetson Orin NX 8GB (Tegra R36.4.7)
- **CPU**: 6x ARM Cortex-A78AE @ 1.73 GHz (aarch64)
- **Memory**: 7.4 GB **unified** (shared CPU + GPU)
- **GPU**: Orin iGPU, 8 SMs, Compute 8.7, ~5.3 TFLOPS BF16
- **Storage**: 116GB eMMC (mmcblk0)
- **CUDA**: 12.6, PyTorch 2.8.0
- **Python**: 3.10.12
- **bf16**: Supported
- **Triton**: Not available (aarch64)
- **Flash Attention 3**: Not available (uses SDPA fallback)
- **Snap browsers**: Broken on Tegra kernel (use `~/.local/bin/brave` launcher)

## Current Production Setup

The Jetson runs two main services:

1. **llama.cpp + Qwen2.5-1.5B-Instruct** — LLM inference server on port 8000
   - Model: `Qwen2.5-1.5B-Instruct-Q4_K_M` (~1.1GB GGUF, ~1.8GB RAM at runtime)
   - Speed: 286 tok/s prompt, 29 tok/s generation
   - Start: `bash scripts/start_llm_server.sh`

2. **ClawMesh** — mesh farm controller with phone UI on port 3000
   - Connects to LLM via OpenAI-compatible API at `http://127.0.0.1:8000/v1`
   - Phone command center: `http://<jetson-ip>:3000/command`
   - Start: `cd /home/jetson/repo/clawmesh && npx tsx clawmesh.ts start --name jetson-field --field-node --pi-planner --pi-model nanochat/d4 --telegram`

## Setup Steps

### 1. Add swap (one-time)
```bash
sudo fallocate -l 4G /var/nanochat_swap
sudo chmod 600 /var/nanochat_swap
sudo mkswap /var/nanochat_swap
sudo swapon /var/nanochat_swap
```

### 2. Drop caches (before training or starting LLM server)
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
Without this, Jetson's nvmap allocator fails with `NvMapMemAllocInternalTagged error 12` (ENOMEM for contiguous GPU pages in unified memory).

### 3. Environment variables
```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DTYPE="bfloat16"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
```

## Code Changes (vs upstream karpathy/nanochat)

1. **`nanochat/optim.py`** — Conditional `torch.compile` (no-op when Triton missing)
2. **`scripts/base_train.py`** — Conditional `torch.compile` via try/except
3. **`scripts/chat_sft.py`** — `--lite` mode, `--load-optimizer=0`, gradient clipping, Bhoomi data integration
4. **`nanochat/gpt.py`** — NaN guard for all-masked batches in bestfit packer
5. **`nanochat/common.py`** — Orin added to peak FLOPS table
6. **`nanochat/dataset.py`** — `NANOCHAT_DATA_DIR` env var for custom data dirs
7. **`scripts/chat_web.py`** — OpenAI-compatible `/v1/chat/completions` endpoint
8. **`scripts/fetch_bhoomi_transcripts.py`** — YouTube transcript pipeline for Bhoomi Natural
9. **`scripts/bhoomi_to_parquet.py`** — Transcript-to-parquet conversion for pretraining
10. **`scripts/bhoomi_sft.py`** — Focused domain SFT for farming knowledge
11. **`scripts/start_llm_server.sh`** — llama.cpp server launcher for Qwen2.5-1.5B

## Bhoomi Natural Farming Dataset

Built from YouTube transcripts of the Bhoomi Natural channel:
- 22 transcripts fetched (Hindi + English), 272K chars (~68K tokens)
- Pretrain data: `~/.cache/nanochat/base_data_bhoomi/` (parquet shards)
- SFT data: `~/.cache/nanochat/bhoomi_data/bhoomi_conversations.jsonl` (69 conversations)

Pipeline:
```bash
python -m scripts.fetch_bhoomi_transcripts   # fetch YouTube transcripts
python -m scripts.bhoomi_to_parquet          # convert to parquet for pretraining
python -m scripts.bhoomi_sft                 # focused SFT on farming data
```

## NanoChat D4 Training Configs (for reference)

### Pretrain
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--depth` | 4 | 36.7M params |
| `--aspect-ratio` | 64 | n_embd=256 |
| `--device-batch-size` | 1 | OOM at 2 |
| `--total-batch-size` | 4096 | 16 grad accum steps |
| `--window-pattern` | L | Full context (no sliding window) |
| Peak memory | 535 MB | |
| Throughput | ~3,200 tok/sec | |

### SFT
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--max-seq-len` | 1024 | Must be >= 1024 |
| `--device-batch-size` | 1 | |
| `--total-batch-size` | 1024 | |
| `--lite` | flag | Skips SmolTalk/MMLU/Spelling |
| `--load-optimizer=0` | flag | Skip optimizer state (~180MB saved) |
| Peak memory | 873 MB | |
| Throughput | ~5,400 tok/sec | |

Note: The NanoChat D4 model (36.7M params) is too small for coherent responses. Production inference now uses Qwen2.5-1.5B via llama.cpp instead.

## File Locations
- NanoChat base data: `~/.cache/nanochat/base_data_climbmix/`
- Bhoomi data: `~/.cache/nanochat/bhoomi_data/`
- Tokenizer: `~/.cache/nanochat/tokenizer.pkl`
- Base checkpoints: `~/.cache/nanochat/base_checkpoints/d4/`
- SFT checkpoints: `~/.cache/nanochat/chatsft_checkpoints/d4/`
- Identity data: `~/.cache/nanochat/identity_conversations.jsonl`
- llama.cpp: `/home/jetson/repo/llama.cpp/`
- GGUF model: `/home/jetson/repo/llama.cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Swap file: `/var/nanochat_swap` (4GB)

## Quick Commands

```bash
# Start LLM server (Qwen2.5 via llama.cpp)
cd /home/jetson/repo/nanochat && bash scripts/start_llm_server.sh

# Start ClawMesh (farm controller + phone UI)
cd /home/jetson/repo/clawmesh && npx tsx clawmesh.ts start \
  --name jetson-field --field-node --sensor-interval 10000 \
  --pi-planner --pi-model nanochat/d4 --telegram

# NanoChat pretrain (D4)
cd /home/jetson/repo/nanochat && source .venv/bin/activate
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
python -m scripts.base_train --depth=4 --aspect-ratio=64 --head-dim=64 --window-pattern=L \
    --max-seq-len=256 --device-batch-size=1 --total-batch-size=4096 --num-iterations=500

# NanoChat SFT (lite)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
python -m scripts.chat_sft --lite --load-optimizer=0 \
    --max-seq-len=1024 --device-batch-size=1 --total-batch-size=1024

# Chat (interactive / web)
python -m scripts.chat_cli
python -m scripts.chat_web
```

## Known Issues
- Snap browsers (Brave, Chromium, Firefox) fail on Tegra kernel due to missing `CONFIG_SECURITY_FILE_CAPABILITIES`. Use `~/.local/bin/brave` launcher that runs Brave directly with `--no-sandbox`.
- SDPA does not support sliding window attention on Orin — use `--window-pattern L`.
- Must drop kernel caches before GPU-intensive work to avoid nvmap allocation failures.
- Close Brave browser before training (~2GB RAM savings).
