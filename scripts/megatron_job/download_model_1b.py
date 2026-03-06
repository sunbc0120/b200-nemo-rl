#!/usr/bin/env python3
import os
import sys
import time

# Ensure HF_HOME is set to the infinite GCS mount BEFORE importing huggingface_hub
os.environ["HF_HOME"] = "/data/huggingface"
# Disable hf_transfer. The Rust backend is notoriously brittle with network drops
# and FUSE backend uploads. The native Python downloader is much more stable.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Please run: pip install huggingface_hub")
    sys.exit(1)

model_id = "google/gemma-3-1b-it"

print(f"Starting robust download of {model_id} to {os.environ['HF_HOME']}...")
print("This may take a while. The native downloader will automatically handle CDN network drops.")

max_retries = 50
for attempt in range(1, max_retries + 1):
    try:
        # Use max_workers to parallelize the download, but keep it reasonable for FUSE
        path = snapshot_download(
            repo_id=model_id,
            max_workers=4,
            resume_download=True,
            local_dir_use_symlinks=False # Best practice for GCSFuse
        )
        print(f"\nSuccess! Model fully downloaded to: {path}")
        break
    except Exception as e:
        print(f"\n[Attempt {attempt}/{max_retries}] Network drop detected: {e}")
        if attempt == max_retries:
            print("Fatal: Download failed after maximum retries.")
            sys.exit(1)
        print("Reconnecting seamlessly in 10 seconds...")
        time.sleep(10)
