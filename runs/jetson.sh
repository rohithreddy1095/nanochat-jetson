#!/bin/bash
set -e

# =============================================================================
# nanochat on Jetson Orin (8GB unified memory)
# =============================================================================
# This script trains a small LLM end-to-end on a Jetson Orin device.
#
# BEFORE RUNNING: Close Brave browser and unnecessary apps to free ~2GB RAM!
#   The Jetson shares 7.4GB between CPU and GPU. Every MB counts.
#
# Usage:
#   bash runs/jetson.sh           # full pipeline
#   bash runs/jetson.sh step2     # run from a specific step (step1..step6)
#
# Optional: Add swap on SD card first (see runs/jetson_swap.sh)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Configuration (edit these to experiment!) ---
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DTYPE="bfloat16"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4  # Jetson has 6 cores

# Model config — the single dial
DEPTH=4                    # transformer depth (4 = ~37M params, fits in 8GB Jetson)
ASPECT_RATIO=64            # model_dim = depth * aspect_ratio (= 256 for d4)
HEAD_DIM=64                # smaller head dim for small models
SEQ_LEN=512                # context length
WINDOW_PATTERN="L"         # L=full context (no sliding window, SDPA compatible)

# Training config
DEVICE_BATCH_SIZE=1        # per-device batch size (1 = minimum, saves memory)
TOTAL_BATCH_SIZE=4096      # total batch in tokens (gradient accumulation handles the rest)
NUM_ITERATIONS=3000        # training steps (increase for better results, decrease for speed)
EVAL_EVERY=200             # evaluate every N steps
EVAL_TOKENS=65536          # tokens for validation eval (small to save time/memory)
SAMPLE_EVERY=500           # sample from model every N steps

# SFT config
SFT_ITERATIONS=1000
SFT_EVAL_EVERY=200

# Tokenizer config
TOK_MAX_CHARS=500000000    # 500M chars for tokenizer training (smaller = faster)
DATA_SHARDS=2              # number of data shards to download (~100MB each)

WANDB_RUN="${WANDB_RUN:-dummy}"

mkdir -p "$NANOCHAT_BASE_DIR"

# --- Helper ---
log() { echo -e "\n\033[1;32m>>> $1\033[0m\n"; }
warn() { echo -e "\033[1;33m⚠  $1\033[0m"; }
meminfo() {
    python3 -c "
import psutil
m = psutil.virtual_memory()
print(f'RAM: {m.used/1024**3:.1f}/{m.total/1024**3:.1f} GB used ({m.percent}%) | available: {m.available/1024**3:.1f} GB')
import torch
if torch.cuda.is_available():
    f,t = torch.cuda.mem_get_info()
    print(f'GPU: {(t-f)/1024**3:.1f}/{t/1024**3:.1f} GB used | free: {f/1024**3:.1f} GB')
" 2>/dev/null || true
}

# --- Preflight checks ---
preflight() {
    log "Preflight checks"
    meminfo

    # CRITICAL for Jetson: drop filesystem caches to free contiguous memory
    # The Jetson nvmap allocator needs contiguous pages for GPU allocations
    log "Dropping filesystem caches to free memory for GPU..."
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || warn "Could not drop caches (need sudo)"
    sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true

    # Add swap if not already present (Jetson's 7.4GB shared RAM is tight)
    if ! swapon --show | grep -q nanochat_swap; then
        log "Adding 4GB swap file for training stability..."
        if [ ! -f /var/nanochat_swap ]; then
            sudo fallocate -l 4G /var/nanochat_swap 2>/dev/null || \
                sudo dd if=/dev/zero of=/var/nanochat_swap bs=1M count=4096 status=progress
            sudo chmod 600 /var/nanochat_swap
            sudo mkswap /var/nanochat_swap
        fi
        sudo swapon /var/nanochat_swap 2>/dev/null || true
    fi

    # Check available memory
    AVAIL_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available/1024**3:.1f}')")
    if (( $(echo "$AVAIL_GB < 2.0" | bc -l) )); then
        warn "Only ${AVAIL_GB}GB RAM available. Close browsers & apps for best results!"
        warn "Brave browser alone uses ~2GB. Run: pkill -f brave"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi

    # Activate venv
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        warn "No .venv found. Creating..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e .
    fi

    log "System ready"
    meminfo
}

# --- Step 1: Download data ---
step1() {
    log "Step 1: Download training data ($DATA_SHARDS shards)"
    python -m nanochat.dataset -n "$DATA_SHARDS"
}

# --- Step 2: Train tokenizer ---
step2() {
    log "Step 2: Train tokenizer (on ${TOK_MAX_CHARS} chars)"
    python -m scripts.tok_train --max-chars="$TOK_MAX_CHARS"
    python -m scripts.tok_eval
}

# --- Step 3: Pretrain base model ---
step3() {
    log "Step 3: Pretrain base model (depth=$DEPTH, ${NUM_ITERATIONS} steps)"
    log "Config: seq_len=$SEQ_LEN, batch=$DEVICE_BATCH_SIZE, total_batch=$TOTAL_BATCH_SIZE"
    meminfo

    python -m scripts.base_train \
        --depth="$DEPTH" \
        --aspect-ratio="$ASPECT_RATIO" \
        --head-dim="$HEAD_DIM" \
        --window-pattern="$WINDOW_PATTERN" \
        --max-seq-len="$SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --num-iterations="$NUM_ITERATIONS" \
        --eval-every="$EVAL_EVERY" \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=-1 \
        --sample-every="$SAMPLE_EVERY" \
        --save-every=-1 \
        --run="$WANDB_RUN"

    log "Step 3 complete! Base model trained."
}

# --- Step 4: Evaluate base model ---
step4() {
    log "Step 4: Evaluate base model"
    python -m scripts.base_eval \
        --device-batch-size=1 \
        --split-tokens=16384 \
        --max-per-task=16
}

# --- Step 5: SFT (supervised fine-tuning for chat) ---
step5() {
    log "Step 5: Supervised fine-tuning (SFT)"

    # Download identity conversations
    IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
    if [ ! -f "$IDENTITY_FILE" ]; then
        curl -L -o "$IDENTITY_FILE" \
            https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi

    python -m scripts.chat_sft \
        --max-seq-len="$SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --eval-every="$SFT_EVAL_EVERY" \
        --eval-tokens="$EVAL_TOKENS" \
        --num-iterations="$SFT_ITERATIONS" \
        --run="$WANDB_RUN"

    log "Step 5 complete! Chat model ready."
}

# --- Step 6: Chat! ---
step6() {
    log "Step 6: Chat with your model!"
    echo "Choose:"
    echo "  cli  - Command-line chat"
    echo "  web  - Web UI (ChatGPT-style)"
    echo "  test - Quick test prompt"
    read -p "Choice [test]: " choice
    choice="${choice:-test}"

    case "$choice" in
        cli)
            python -m scripts.chat_cli
            ;;
        web)
            echo "Starting web UI... visit http://$(hostname -I | awk '{print $1}'):8000/"
            python -m scripts.chat_web
            ;;
        test|*)
            python -m scripts.chat_cli -p "What is the capital of France?"
            python -m scripts.chat_cli -p "Why is the sky blue?"
            ;;
    esac
}

# --- Main ---
START_STEP="${1:-all}"

preflight

case "$START_STEP" in
    step1) step1 ;;
    step2) step2 ;;
    step3) step3 ;;
    step4) step4 ;;
    step5) step5 ;;
    step6) step6 ;;
    all)
        step1
        step2
        step3
        step4
        step5
        step6
        ;;
    *)
        echo "Usage: $0 [all|step1|step2|step3|step4|step5|step6]"
        exit 1
        ;;
esac

log "Done! 🎉"
