#!/usr/bin/env python3
"""
Convert Bhoomi Natural transcripts into parquet shards for NanoChat pretraining.

Reads the pretrain_corpus.txt from fetch_bhoomi_transcripts and creates
parquet shards in the same format as the ClimbMix dataset.

Usage:
  python -m scripts.bhoomi_to_parquet
  python -m scripts.bhoomi_to_parquet --output-dir /path/to/output
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


def main():
    parser = argparse.ArgumentParser(description="Convert Bhoomi transcripts to parquet shards")
    parser.add_argument("--input", default=None, help="Input pretrain_corpus.txt path")
    parser.add_argument("--output-dir", default=None, help="Output directory for parquet shards")
    args = parser.parse_args()

    base_dir = get_base_dir()
    input_path = args.input or os.path.join(base_dir, "bhoomi_data", "pretrain_corpus.txt")
    output_dir = args.output_dir or os.path.join(base_dir, "base_data_bhoomi")
    os.makedirs(output_dir, exist_ok=True)

    # Read and split into documents
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    docs = [d.strip() for d in raw.split("\n\n---\n\n") if d.strip()]
    print(f"Loaded {len(docs)} documents from {input_path}")
    total_chars = sum(len(d) for d in docs)
    print(f"Total characters: {total_chars:,}")

    if not docs:
        print("No documents found, exiting.")
        return

    # Write train shard (all but last doc) and val shard (last doc)
    # Use same format as repackage_data_reference.py
    row_group_size = 1024

    # For small datasets, put everything in one train shard + one val shard
    if len(docs) <= 2:
        train_docs = docs[:1]
        val_docs = docs[1:] if len(docs) > 1 else docs[:1]
    else:
        val_docs = docs[-1:]
        train_docs = docs[:-1]

    # Duplicate train docs to make training more effective (multiple epochs in data)
    # For ~68K tokens this helps the model see the data enough times
    epoch_multiplier = max(1, 500_000 // max(total_chars, 1))  # target ~500K chars
    if epoch_multiplier > 1:
        print(f"Duplicating train docs {epoch_multiplier}x to reach ~500K chars")
        train_docs = train_docs * epoch_multiplier

    # Write train shard
    train_path = os.path.join(output_dir, "shard_00000.parquet")
    train_table = pa.Table.from_pydict({"text": train_docs})
    pq.write_table(
        train_table, train_path,
        row_group_size=min(row_group_size, len(train_docs)),
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    print(f"Wrote train shard: {train_path} ({len(train_docs)} docs)")

    # Write val shard (always the last shard by convention)
    val_path = os.path.join(output_dir, "shard_00001.parquet")
    val_table = pa.Table.from_pydict({"text": val_docs})
    pq.write_table(
        val_table, val_path,
        row_group_size=min(row_group_size, len(val_docs)),
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    print(f"Wrote val shard: {val_path} ({len(val_docs)} docs)")

    print(f"\nDone! Parquet shards in: {output_dir}")
    print(f"To pretrain: NANOCHAT_DATA_DIR={output_dir} python -m scripts.base_train --depth 4")


if __name__ == "__main__":
    main()
