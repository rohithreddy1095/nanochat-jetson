#!/usr/bin/env python3
"""
Quick Bhoomi-only SFT on top of an existing chat model.
Focused domain adaptation for natural farming knowledge.

Usage:
  python -m scripts.bhoomi_sft
  python -m scripts.bhoomi_sft --base-tag d4 --epochs 20
"""

import argparse
import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb

from tasks.common import TaskMixture
from tasks.customjson import CustomJSON


def main():
    parser = argparse.ArgumentParser(description="Bhoomi Natural domain SFT")
    parser.add_argument("--base-tag", default="d4", help="Base SFT model tag to fine-tune")
    parser.add_argument("--output-tag", default=None, help="Output model tag (default: {base-tag}-bhoomi)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of data epochs")
    parser.add_argument("--device-batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    output_tag = args.output_tag or f"{args.base_tag}-bhoomi"

    # Init
    device_type = autodetect_device_type()
    device = torch.device(f"{device_type}:0")
    torch.set_float32_matmul_precision("high")

    base_dir = get_base_dir()
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    # Load base chat model
    model, _, meta_data = load_model("sft", device=device, phase="train", model_tag=args.base_tag)
    model.train()
    model_config = meta_data.get("model_config", {})
    print0(f"Loaded base model: {args.base_tag} (config: {model_config})")

    # Load Bhoomi conversations
    bhoomi_path = os.path.join(base_dir, "bhoomi_data", "bhoomi_conversations.jsonl")
    if not os.path.exists(bhoomi_path):
        print0(f"ERROR: {bhoomi_path} not found. Run fetch_bhoomi_transcripts first.")
        return

    # Also include identity data to prevent catastrophic forgetting
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
    rohith_path = os.path.join(base_dir, "rohith_conversations.jsonl")

    train_tasks = [CustomJSON(filepath=bhoomi_path)] * args.epochs
    if os.path.exists(rohith_path):
        train_tasks.extend([CustomJSON(filepath=rohith_path)] * 5)
    if os.path.exists(identity_path):
        train_tasks.append(CustomJSON(filepath=identity_path))

    dataset = TaskMixture(train_tasks)
    print0(f"Training data: {len(dataset):,} rows ({args.epochs} epochs of Bhoomi + identity)")

    # Tokenize and pack data
    T = args.max_seq_len
    B = args.device_batch_size
    bos_id = tokenizer.get_bos_token_id()

    # Simple optimizer - just AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    step = 0
    total_loss = 0
    t0 = time.time()
    num_rows = len(dataset)

    print0(f"Starting Bhoomi SFT: {num_rows} rows, lr={args.lr}")
    print0(f"Seq len: {T}, Batch size: {B}")
    print0("-" * 60)

    indices = list(range(num_rows))
    import random
    random.shuffle(indices)

    for idx in indices:
        conversation = dataset[idx]
        ids_list, mask_list = tokenizer.render_conversation(conversation)
        if ids_list is None or len(ids_list) < 4:
            continue

        ids_tensor = torch.tensor(ids_list, dtype=torch.long)
        mask_tensor = torch.tensor(mask_list, dtype=torch.float)

        # Truncate to max_seq_len + 1 (for input/target split)
        ids = ids_tensor[:T + 1].to(device).unsqueeze(0)  # (1, T+1)
        mask = mask_tensor[:T + 1].to(device).unsqueeze(0)  # (1, T+1)

        if ids.shape[1] < 4:
            continue

        x = ids[:, :-1]  # input
        y = ids[:, 1:]   # target
        m = mask[:, 1:]  # mask for target tokens

        # Skip if no assistant tokens to train on
        if m.sum() == 0:
            continue

        # Forward
        with torch.amp.autocast(device_type=device_type, dtype=COMPUTE_DTYPE):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.reshape(-1),
                reduction='none'
            )
            # Apply mask: only train on assistant tokens
            loss = (loss * m.reshape(-1)).sum() / m.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 50 == 0:
            avg_loss = total_loss / 50
            dt = time.time() - t0
            print0(f"step {step:05d}/{num_rows} | loss: {avg_loss:.4f} | dt: {dt:.1f}s")
            total_loss = 0
            t0 = time.time()

    # Save
    print0(f"\nTraining complete: {step} steps")
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        optimizer.state_dict(),
        {
            "step": step,
            "val_bpb": None,
            "model_config": model_config,
        },
    )
    print0(f"Saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
