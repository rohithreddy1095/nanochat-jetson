#!/usr/bin/env python3
"""
Fetch YouTube transcripts from Bhoomi Natural channel and prepare them
for NanoChat pretraining and SFT.

Outputs:
  1. Pretraining corpus: Plain text documents (one per video) for base model training
  2. SFT conversations: Q&A pairs about natural farming topics

Usage:
  python -m scripts.fetch_bhoomi_transcripts
  python -m scripts.fetch_bhoomi_transcripts --channel @bhoominatural
  python -m scripts.fetch_bhoomi_transcripts --from-file /tmp/bhoomi_video_ids.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from nanochat.common import get_base_dir


def get_video_list(channel_handle: str) -> list[dict]:
    """Get all video IDs and titles from a YouTube channel."""
    print(f"Fetching video list from {channel_handle}...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", f"https://www.youtube.com/{channel_handle}"],
        capture_output=True, text=True, timeout=60,
    )
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            vid_id, title = line.split("\t", 1)
            videos.append({"id": vid_id.strip(), "title": title.strip()})
    print(f"Found {len(videos)} videos")
    return videos


def get_video_list_from_file(filepath: str) -> list[dict]:
    """Load video IDs from a file (one per line)."""
    videos = []
    with open(filepath) as f:
        for line in f:
            vid_id = line.strip()
            if vid_id:
                videos.append({"id": vid_id, "title": ""})
    print(f"Loaded {len(videos)} video IDs from {filepath}")
    return videos


def fetch_transcript(video_id: str) -> str | None:
    """Fetch transcript for a YouTube video. Tries Hindi first, then English, then auto-generated."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt = YouTubeTranscriptApi()

        # Try fetching directly with language preference (hi > en)
        try:
            entries = ytt.fetch(video_id, languages=["hi", "en"])
            texts = [entry.text for entry in entries]
            return " ".join(texts)
        except Exception:
            pass

        # Fall back: list available transcripts and try auto-generated
        try:
            transcript_list = ytt.list(video_id)
            for transcript in transcript_list:
                if transcript.language_code in ("hi", "en"):
                    entries = transcript.fetch()
                    texts = [entry.text for entry in entries]
                    return " ".join(texts)
        except Exception:
            pass

        return None

    except Exception as e:
        print(f"  [skip] {video_id}: {e}")
        return None


def clean_transcript(text: str) -> str:
    """Clean up transcript text."""
    # Remove [Music], [Applause] etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove very short fragments
    if len(text) < 100:
        return ""
    return text


def transcript_to_document(title: str, text: str) -> str:
    """Convert a transcript into a pretraining document."""
    return f"# {title}\n\n{text}"


def transcript_to_conversations(title: str, text: str) -> list[list[dict]]:
    """Convert a transcript into SFT conversation pairs."""
    conversations = []

    # Split transcript into chunks of ~500 chars for digestible Q&A
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        current_chunk.append(sent)
        current_len += len(sent)
        if current_len > 400:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Create a summary conversation for the whole video
    if len(text) > 200:
        conversations.append([
            {"role": "user", "content": f"What does the Bhoomi Natural video \"{title}\" talk about?"},
            {"role": "assistant", "content": text[:800] + ("..." if len(text) > 800 else "")},
        ])

    # Create topic-specific Q&A from chunks
    for i, chunk in enumerate(chunks[:6]):  # Max 6 Q&A per video
        if len(chunk) < 80:
            continue

        # Extract likely topic from the chunk
        topic = title.split("|")[0].strip().split(":")[0].strip()

        conversations.append([
            {"role": "user", "content": f"Tell me about {topic} - part {i+1}"},
            {"role": "assistant", "content": chunk},
        ])

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Fetch Bhoomi Natural YouTube transcripts")
    parser.add_argument("--channel", default="@bhoominatural", help="YouTube channel handle")
    parser.add_argument("--from-file", default=None, help="Load video IDs from file (one per line)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    output_dir = args.output_dir or os.path.join(base_dir, "bhoomi_data")
    os.makedirs(output_dir, exist_ok=True)

    # Get video list
    if args.from_file:
        videos = get_video_list_from_file(args.from_file)
        # Fetch titles for videos that don't have them
        print("Fetching video titles...")
        titled_videos = get_video_list(args.channel)
        title_map = {v["id"]: v["title"] for v in titled_videos}
        for v in videos:
            if not v["title"]:
                v["title"] = title_map.get(v["id"], f"Bhoomi Natural Video {v['id']}")
    else:
        videos = get_video_list(args.channel)

    # Fetch transcripts
    print(f"\nFetching transcripts for {len(videos)} videos...")
    transcripts = []
    failed = []

    for i, video in enumerate(videos):
        print(f"  [{i+1}/{len(videos)}] {video['title'][:60]}...", end=" ", flush=True)

        raw = fetch_transcript(video["id"])
        if raw:
            cleaned = clean_transcript(raw)
            if cleaned:
                transcripts.append({
                    "id": video["id"],
                    "title": video["title"],
                    "text": cleaned,
                    "char_count": len(cleaned),
                })
                print(f"OK ({len(cleaned)} chars)")
            else:
                failed.append(video["id"])
                print("too short")
        else:
            failed.append(video["id"])
            print("no transcript")

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\nResults: {len(transcripts)} transcripts, {len(failed)} failed")
    total_chars = sum(t["char_count"] for t in transcripts)
    print(f"Total corpus size: {total_chars:,} characters (~{total_chars // 4:,} tokens)")

    # Save raw transcripts
    raw_path = os.path.join(output_dir, "transcripts.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nSaved raw transcripts to {raw_path}")

    # Generate pretraining corpus (plain text documents)
    pretrain_path = os.path.join(output_dir, "pretrain_corpus.txt")
    with open(pretrain_path, "w", encoding="utf-8") as f:
        for t in transcripts:
            doc = transcript_to_document(t["title"], t["text"])
            f.write(doc + "\n\n---\n\n")
    print(f"Saved pretraining corpus to {pretrain_path}")

    # Generate SFT conversations
    all_convos = []
    for t in transcripts:
        convos = transcript_to_conversations(t["title"], t["text"])
        all_convos.extend(convos)

    sft_path = os.path.join(output_dir, "bhoomi_conversations.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for convo in all_convos:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_convos)} SFT conversations to {sft_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BHOOMI NATURAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Videos processed:    {len(transcripts)}/{len(videos)}")
    print(f"Total characters:    {total_chars:,}")
    print(f"Estimated tokens:    ~{total_chars // 4:,}")
    print(f"SFT conversations:   {len(all_convos)}")
    print(f"{'='*60}")
    print(f"\nPretraining corpus:  {pretrain_path}")
    print(f"SFT conversations:   {sft_path}")
    print(f"Raw transcripts:     {raw_path}")
    print(f"\nNext steps:")
    print(f"  1. Generate parquet shard:  python -m scripts.bhoomi_to_parquet")
    print(f"  2. Retrain tokenizer:       python -m scripts.tok_train  (include bhoomi data)")
    print(f"  3. Pretrain with bhoomi:    python -m scripts.pretrain --depth 4")
    print(f"  4. SFT with bhoomi:         python -m scripts.chat_sft --bhoomi")


if __name__ == "__main__":
    main()
