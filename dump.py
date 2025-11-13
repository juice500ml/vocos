#!/usr/bin/env python3
"""
Script to pre-compute and cache SSL features for all audio files in a filelist.
This allows features to be cached before training, avoiding recalculation.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torch import no_grad


def dump_features_for_filelist(
    filelist_path: str,
    cache_dir: str,
    ssl_model: str = "microsoft/wavlm-large",
    layer: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Pre-compute and cache SSL features for all files in the filelist.
    Audio is always resampled to 16kHz.

    Args:
        filelist_path: Path to filelist text file
        cache_dir: Directory to store cached features
        ssl_model: SSL model name
        layer: Layer index to extract (-1 for last layer)
        device: Device to run feature extraction on
    """
    sampling_rate = 16000  # Always 16kHz
    # Read filelist
    with open(filelist_path, 'r') as f:
        filelist = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Processing {len(filelist)} files from {filelist_path}")
    print(f"SSL Model: {ssl_model}, Layer: {layer}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {device}")
    print("-" * 80)

    # Create cache directory
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    print("Loading SSL model...")
    model = AutoModel.from_pretrained(ssl_model).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model)
    model.eval()

    # Generate cache key function
    import hashlib
    def get_cache_key(file_path: str) -> str:
        abs_path = os.path.abspath(file_path)
        return hashlib.md5(abs_path.encode()).hexdigest()

    def get_cache_path(cache_key: str) -> Path:
        return cache_dir / f"{cache_key}.pt"

    # Metadata file to store audio_length mappings
    metadata_path = cache_dir / "meta.json"

    # Load existing metadata if available
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")

    # Process each file
    cached_count = 0
    processed_count = 0
    error_count = 0

    for audio_path in tqdm(filelist, desc="Processing files"):
        abs_path = os.path.abspath(audio_path)

        # Check if already cached
        cache_key = get_cache_key(audio_path)
        cache_path = get_cache_path(cache_key)

        if cache_path.exists():
            cached_count += 1
            continue

        try:
            # Load audio file
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                error_count += 1
                continue

            y, sr = torchaudio.load(audio_path)

            # Check if empty
            if y.numel() == 0:
                print(f"Warning: Empty audio file: {audio_path}")
                error_count += 1
                continue

            # Mix to mono if needed
            if y.size(0) > 1:
                y = y.mean(dim=0, keepdim=True)

            # Resample to target sampling rate if needed
            if sr != sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sampling_rate)

            # Convert to numpy for processor
            audio_np = y.squeeze().cpu().numpy()

            # Process with SSL model
            with no_grad():
                x = processor(
                    raw_speech=audio_np,
                    sampling_rate=sampling_rate,
                    padding=True,
                    return_tensors="pt"
                )
                x = {k: t.to(device) for k, t in x.items()}
                
                if layer == -1:
                    outputs = model(**x)
                    features = outputs.last_hidden_state.detach()[0]
                else:
                    outputs = model(output_hidden_states=True, **x)
                    features = outputs.hidden_states[layer].detach()[0]

            # Save to cache directly (no dict, no transpose)
            # Features shape: (batch=1, seq_len, hidden_dim)
            torch.save(features.cpu(), cache_path)
            metadata[cache_key] = {"length": features.size(0), "audio_path": abs_path}
            processed_count += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            error_count += 1
            continue

    # Save metadata file
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")

    # Summary
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Total files: {len(filelist)}")
    print(f"  Already cached: {cached_count}")
    print(f"  Newly processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Metadata entries: {len(metadata)}")
    print(f"  Cache directory: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute and cache SSL features for audio files")
    parser.add_argument("filelist_path", type=str, help="Path to filelist text file")
    parser.add_argument("cache_dir", type=str, help="Directory to store cached features")
    parser.add_argument("--ssl_model", type=str, default="microsoft/wavlm-large",
                       help="SSL model name (default: microsoft/wavlm-large)")
    parser.add_argument("--layer", type=int, default=-1,
                       help="Layer index to extract (-1 for last layer, default: -1)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: cuda if available, else cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.filelist_path):
        print(f"Error: Filelist not found: {args.filelist_path}")
        return

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    dump_features_for_filelist(
        filelist_path=args.filelist_path,
        cache_dir=args.cache_dir,
        ssl_model=args.ssl_model,
        layer=args.layer,
        device=args.device,
    )


if __name__ == "__main__":
    main()
