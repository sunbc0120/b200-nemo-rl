import os
import json
import glob
import shutil
import argparse
from transformers import AutoConfig, AutoTokenizer
import yaml

def main():
    parser = argparse.ArgumentParser(description="Convert FSDP safetensors checkpoint to HF checkpoint natively via GCS cache")
    parser.add_argument("--step-dir", type=str, required=True, help="Path to the specific step directory (e.g., /data/nemo-rl-results/grpo-gemma3.../step_30)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the unified HF checkpoint")
    args = parser.parse_args()

    model_dir = os.path.join(args.step_dir, "policy/weights/model")
    config_path = os.path.join(args.step_dir, "config.yaml")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Copying safetensors (this may take a minute)...")
    for sf in glob.glob(os.path.join(model_dir, "*.safetensors")):
        basename = os.path.basename(sf)
        dest = os.path.join(args.output_dir, basename)
        if not os.path.exists(dest):
            shutil.copy(sf, dest)

    print("Building index mapping...")
    with open(os.path.join(model_dir, ".hf_metadata/fqn_to_file_index_mapping.json")) as f:
        mapping = json.load(f)

    index_to_file = {}
    for sf in glob.glob(os.path.join(model_dir, "*.safetensors")):
        basename = os.path.basename(sf)
        idx = int(basename.split('-')[1])
        index_to_file[idx] = basename

    weight_map = {k: index_to_file[v] for k, v in mapping.items()}

    index_data = {
        "metadata": {"total_size": sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in index_to_file.values() if os.path.exists(os.path.join(args.output_dir, f)))},
        "weight_map": weight_map
    }

    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print("Copying base model configuration from LOCAL CACHE...")
    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)
    
    # Try to grab the base model to find the matching cached tokenizer
    base_model = config_yaml.get("policy", {}).get("model_name", "google/gemma-3-1b-it")

    # Setting local_files_only=True forces HF to look in the HF_HOME cache instead of the internet!
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    config.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"HF Merge Successful! Location: {args.output_dir}")

if __name__ == "__main__":
    main()
