# Jetson Orin Nanochat Storyboard

Timestamp: 2025-11-15T16:03:43.537Z

## 1. Environment Bootstrapping
```bash
cd /home/jetson/repos/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" && mkdir -p "$NANOCHAT_BASE_DIR"
export PATH="/usr/local/cuda-12.6/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv && source .venv/bin/activate
uv sync --extra cpu
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```
*Sets deterministic threading, prepares cache directories, adds CUDA 12.6 binaries (nvcc) to PATH, installs uv deps (CPU Torch wheel), bootstraps Rust, and builds the Rust BPE Python extension.*

## 2. Reset Prior Artifacts
```bash
python -m nanochat.report reset
```
*Clears `report.md` plus cached metrics so the upcoming run records fresh telemetry.*

## 3. Sample Data & Tokenizer
1. ```bash
   python -m nanochat.dataset -n 1
   ```
   *Downloads one shard (~250M chars) into `$NANOCHAT_BASE_DIR/raw_data`, tokenizes into `.npy` chunks, and updates the registry.*
2. ```bash
   python -m scripts.tok_train --max_chars=5_000_000
   ```
   *Feeds 5M chars into the Rust BPE trainer, creating tokenizer model/vocab artifacts.*
3. ```bash
   python -m scripts.tok_eval
   ```
   *Runs tokenizer quality checks (NLL, byte coverage) to validate the BPE before training.*

## 4. Base Pretraining (Toy Config)
```bash
python -m scripts.base_train \
  --device_type=cuda \
  --depth=2 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --total_batch_size=512 \
  --num_iterations=30 \
  --eval_every=10 \
  --eval_tokens=2048 \
  --core_metric_every=-1 \
  --sample_every=30
```
*Initializes a depth-2 GPT, builds tokenized dataloaders, configures grad accumulation to keep memory tiny, and runs 30 steps while writing checkpoints to `$NANOCHAT_BASE_DIR/base_checkpoints/d2`.*

## 5. Base Diagnostics
```bash
python -m scripts.base_loss --device_batch_size=1 --split_tokens=2048
python -m scripts.base_eval --max-per-task=8
```
*First command reloads the base checkpoint to compute validation bits-per-byte; second runs the CORE mini-benchmark via `nanochat.engine.Engine`.*

## 6. Midtraining
```bash
python -m scripts.mid_train \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --total_batch_size=512 \
  --eval_every=10 \
  --eval_tokens=2048 \
  --num_iterations=40
```
*Continues training on instruction/mid datasets using the base checkpoint, logging results under `$NANOCHAT_BASE_DIR/mid_checkpoints/d2`.*

## 7. Mid-Stage Evaluation
```bash
python -m scripts.chat_eval --source=mid --max-new-tokens=64 --max-problems=10
```
*Streams inference through the mid model to auto-solve small prompt sets, verifying sampling and KV-cache behavior.*

## 8. SFT (Chat Finetune)
```bash
python -m scripts.chat_sft \
  --device_batch_size=1 \
  --target_examples_per_step=2 \
  --num_iterations=40 \
  --eval_steps=5 \
  --eval_metrics_max_problems=8
```
*Loads the mid checkpoint, fine-tunes on curated chat data, and periodically measures alignment metrics before saving to `$NANOCHAT_BASE_DIR/chat_checkpoints`.*

## 9. Report & Qualitative Checks
```bash
python -m nanochat.report generate
python -m scripts.chat_cli -p "Explain why the sky looks blue."
python -m scripts.chat_web
```
*Generates `report.md`, sanity-checks the final model via CLI streaming, and launches the FastAPI/uvicorn web UI (visit `http://<jetson-ip>:8000/`) to complete the end-to-end workflow.*
