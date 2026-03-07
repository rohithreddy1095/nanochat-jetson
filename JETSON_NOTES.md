# Nanochat on Jetson Orin — Learnings & Reference

## System Specs
- **Device**: NVIDIA Jetson Orin NX 8GB (Tegra R36.4.7)
- **CPU**: 6× ARM Cortex-A78AE @ 1.73 GHz (aarch64)
- **Memory**: 7.4 GB **unified** (shared CPU + GPU)
- **GPU**: Orin iGPU, 8 SMs, Compute 8.7, ~5.3 TFLOPS BF16
- **Storage**: 116GB eMMC (mmcblk0), ~70GB free
- **CUDA**: 12.6, PyTorch 2.8.0
- **Python**: 3.10.12
- **bf16**: ✅ Supported
- **Triton**: ❌ Not available (aarch64)
- **Flash Attention 3**: ❌ Not available (uses SDPA fallback)
- **sudo password**: jetson_1095

## Critical Setup Steps (MUST DO before training)

### 1. Add swap (one-time)
```bash
echo "jetson_1095" | sudo -S fallocate -l 4G /var/nanochat_swap
echo "jetson_1095" | sudo -S chmod 600 /var/nanochat_swap
echo "jetson_1095" | sudo -S mkswap /var/nanochat_swap
echo "jetson_1095" | sudo -S swapon /var/nanochat_swap
```

### 2. Kill Brave browser (~2GB RAM savings)
```bash
pkill -9 brave
```

### 3. Drop caches (before EVERY training run)
```bash
echo "jetson_1095" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
Without this, Jetson's nvmap allocator fails with `NvMapMemAllocInternalTagged error 12` (ENOMEM for contiguous GPU pages in unified memory).

### 4. Environment variables
```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DTYPE="bfloat16"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
```

## Code Changes Made (vs upstream nanochat)

### 1. `nanochat/optim.py` — Conditional torch.compile
Triton is not available on aarch64/Jetson. Changed `@torch.compile(dynamic=False, fullgraph=True)` to use a conditional decorator `_compile_decorator` that becomes a no-op when Triton is missing.

### 2. `scripts/base_train.py` — Conditional torch.compile
Wrapped `model = torch.compile(model, dynamic=False)` in a try/except for Triton import.

### 3. `scripts/chat_sft.py` — Conditional torch.compile + gradient clipping + NaN guard
- Same torch.compile conditional pattern
- Added `--lite` mode for lightweight SFT on Jetson (skips SmolTalk/MMLU/Spelling)
- Added `--load-optimizer=0` to skip optimizer state loading (saves ~180MB)
- Added gradient clipping (`clip_grad_norm_(max_norm=1.0)`) to prevent gradient explosions
- Logs gradient norm in training output

### 4. `nanochat/gpt.py` — NaN guard for all-masked batches
The bestfit packer can create batches where ALL targets are masked (-1), causing `cross_entropy(reduction='mean')` to return NaN (0/0). Added guard: when no valid targets exist, returns zero loss with grad_fn.

### 5. `nanochat/common.py` — Added Orin to peak FLOPS table
Added `(["orin"], 5.3e12)` to `_PEAK_FLOPS_TABLE` for MFU reporting.

## Pretrain Configuration (d4)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--depth` | 4 | 36.7M params (25M embeddings, 3.1M transformer) |
| `--aspect-ratio` | 64 | → n_embd=256 |
| `--head-dim` | 64 | → 4 heads |
| `--max-seq-len` | 256 | |
| `--device-batch-size` | 1 | Must be 1 (OOM at 2) |
| `--total-batch-size` | 4096 | 16 grad accum steps |
| `--window-pattern` | L | Full context (SDPA has no sliding window) |
| `--num-iterations` | 500 | ~10 min training |
| Peak memory | 535 MB | |
| Throughput | ~3,200 tok/sec | |
| MFU | ~4.5% | |
| Final val BPB | 1.882 | |

## SFT Configuration (working!)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--max-seq-len` | 1024 | **Must be ≥1024** — identity convos avg 526 tokens, max 994 |
| `--device-batch-size` | 1 | |
| `--total-batch-size` | 1024 | 1 grad accum step (seq_len=1024) |
| `--embedding-lr` | 0.003 | 10x lower than pretrain's 0.3 |
| `--unembedding-lr` | 0.0003 | ~13x lower than pretrain's 0.004 |
| `--matrix-lr` | 0.0003 | ~67x lower than pretrain's 0.02 |
| `--warmdown-ratio` | 0.3 | |
| `--warmup-ratio` | 0.05 | |
| `--init-lr-frac` | 1.0 | |
| `--final-lr-frac` | 0.1 | |
| `--lite` | (flag) | Skips SmolTalk/MMLU/Spelling |
| `--load-optimizer=0` | (flag) | Skip optimizer state load |
| `--gsm8k-epochs` | 1 | |
| Peak memory | 873 MB | |
| Throughput | ~5,400 tok/sec | |
| MFU | ~7.5% | |
| Training time | 10.4 minutes | 3,146 steps |
| Final val BPB | 1.376 (from 2.88 start, pretrain was 1.88) |

### SFT Bugs Fixed
1. **NaN from all-masked batches**: Bestfit packer + loss mask can produce batches with ALL targets=-1. `cross_entropy(mean)` returns NaN. Fixed in `gpt.py`.
2. **max_seq_len too short**: With seq_len=256, 90.9% of identity conversations (avg 526 tokens) can never be packed. They clog the buffer, causing all-padding batches. Fixed by using seq_len=1024.
3. **LR too high for SFT**: Pretrained LRs (embedding=0.3, matrix=0.02) cause instant NaN. Must use 10-100x lower LRs.
4. **Gradient clipping needed**: Without `clip_grad_norm_(1.0)`, occasional gradient spikes can destabilize training.

## Performance Notes
- Pretrain: ~1.2s per step, ~3,200 tok/sec
- SFT: ~0.19s per step, ~5,400 tok/sec (longer sequences amortize overhead)
- Loss: pretrain 10.4→6.2, SFT 6.x→2.0
- Browser (Brave) eats ~2GB RAM — close before training
- Total system needs ~2.5GB free to train reliably
- Drop caches before every training run

## File Locations
- Base data: `~/.cache/nanochat/base_data_climbmix/` (2 shards + 1 val)
- Tokenizer: `~/.cache/nanochat/tokenizer.pkl`, `token_bytes.pt`
- Base checkpoints: `~/.cache/nanochat/base_checkpoints/d4/`
- SFT checkpoints: `~/.cache/nanochat/chatsft_checkpoints/d4/`
- Identity data: `~/.cache/nanochat/identity_conversations.jsonl` (1000 convos)
- Rohith data: `~/.cache/nanochat/rohith_conversations.jsonl` (140 convos)
- Swap file: `/var/nanochat_swap` (4GB)

## Quick Commands

```bash
# Full pipeline
cd /home/jetson/repo/nanochat && source .venv/bin/activate
bash runs/jetson.sh

# Just pretrain (after setup)
echo "jetson_1095" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
python -m scripts.base_train --depth=4 --aspect-ratio=64 --head-dim=64 --window-pattern=L \
    --max-seq-len=256 --device-batch-size=1 --total-batch-size=4096 --num-iterations=500 \
    --eval-every=100 --eval-tokens=16384 --core-metric-every=-1 --sample-every=250 \
    --save-every=-1 --run=dummy

# SFT (working command!)
echo "jetson_1095" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
python -m scripts.chat_sft --lite --load-optimizer=0 \
    --max-seq-len=1024 --device-batch-size=1 --total-batch-size=1024 \
    --eval-every=100 --eval-tokens=8192 --chatcore-every=-1 --num-iterations=-1 \
    --gsm8k-epochs=1 \
    --embedding-lr=0.003 --unembedding-lr=0.0003 --matrix-lr=0.0003 \
    --warmdown-ratio=0.3 --warmup-ratio=0.05 --init-lr-frac=1.0 --final-lr-frac=0.1 \
    --run=dummy

# Chat (interactive)
python -m scripts.chat_cli

# Chat (web UI)
python -m scripts.chat_web
```
