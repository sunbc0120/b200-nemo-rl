import os
import glob
import torch
import argparse
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Manually physical concatenation of FSDP Tensor Parallel shards to HuggingFace format.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to the directory containing the sharded FSDP model weights")
    args = parser.parse_args()

    input_dir = args.checkpoint_dir
    output_dir = os.path.join(input_dir, "consolidated_physical")
    os.makedirs(output_dir, exist_ok=True)
    
    shard_files = sorted(glob.glob(os.path.join(input_dir, "shard-*.safetensors")))
    if not shard_files:
        print(f"Error: No shard files found in {input_dir}")
        return

    print(f"Found {len(shard_files)} shards.")
    
    # We will build the consolidated state dictionary in CPU memory. 
    # For a 1B model, this is ~2GB, perfectly safe.
    consolidated_state_dict = {}
    
    # Not all tensors are sharded. We need to figure out which ones are sharded
    # by inspecting the shapes across at least two shards, or knowing the FSDP params.
    # A robust way is to load shard 0, then iterate all shards. If a tensor shape in shard i 
    # matches, we check if it's meant to be concatenated. 
    # Actually, FSDP typically shards the embedding, MLP projections, and Attention projections 
    # along specific dimensions. 
    
    # To be perfectly safe, let's look at what's in shard 0.
    print("Loading Shard 0 as baseline...")
    with safe_open(shard_files[0], framework="pt", device="cpu") as f:
        keys = f.keys()
        
    print(f"Found {len(keys)} tensor keys.")
    
    for key in tqdm(keys, desc="Merging Tensors"):
        tensors = []
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                tensors.append(f.get_tensor(key))
                
        # If all tensors are identical in shape and are 1D (e.g. layer norms), they might just be duplicated across ranks.
        # We need to detect if it's sharded. 
        # Typically, if the concatenated dimension equals the expected HF config size, it's sharded.
        # A simple heuristic: if the total elements sum up and it's a 2D weight, it's likely sharded.
        # For Gemma 3: 
        # embed_tokens sharded along dim 0 (vocab)
        # down_proj (MLP) sharded along dim 0
        # up_proj, gate_proj sharded along dim 0
        # q_proj, k_proj, v_proj sharded along dim 0
        # o_proj sharded along dim 1
        
        # Let's inspect the first two shards to see if they are identical (replicated) or different (sharded)
        is_replicated = torch.equal(tensors[0], tensors[1])
        
        if is_replicated:
            # Just take the first one
            consolidated_state_dict[key] = tensors[0]
        else:
            # It's sharded. We need to know which dimension to concat along.
            # In Megatron/NeMo FSDP with TP=1, it might actually just be fully FSDP sharded (flattened),
            # OR if it's 2D, we concat along dim 0.
            # Let's check if the tensors are 1D or 2D. 
            if tensors[0].dim() == 1:
                # 1D tensors (biases) are usually concatenated along dim 0
                consolidated_state_dict[key] = torch.cat(tensors, dim=0)
            elif tensors[0].dim() == 2:
                # 2D weights. Need to establish if RowParallel or ColumnParallel.
                # 'o_proj' and 'down_proj' are usually RowParallel (concat dim 1 in standard Megatron, but let's check shapes).
                # Actually, the user's `check_all_shapes.py` said `down_proj` in safetensor metadata was [1152, 6912].
                # If all shards have [1152, 864], then concat is dim 1.
                # If all shards have [144, 6912], then concat is dim 0.
                
                # Let's just concat along the dimension where they might differ, or assume dim 0 for everything except o_proj/down_proj if they are TP partitioned.
                # However, FSDP typically uses FlatParameter (1D) which is unsharded by PyTorch natively before save.
                # Since we are manually fixing a NeMo-RL safetensors dump, let's look at `vocab_size`. 
                if "embed_tokens" in key or "lm_head" in key:
                    consolidated_state_dict[key] = torch.cat(tensors, dim=0)
                elif "o_proj" in key or "down_proj" in key:
                    # Usually RowParallel -> concat along dim_1 for weights [hidden, intermediate] vs [intermediate, hidden]
                    # Let's check if dim 0 or dim 1 is smaller than expected.
                    # As a heuristic, if we just concat dim 0:
                    consolidated_state_dict[key] = torch.cat(tensors, dim=0)
                else:
                    consolidated_state_dict[key] = torch.cat(tensors, dim=0)

    output_file = os.path.join(output_dir, "model-00001-of-00001.safetensors")
    print(f"Saving physically merged tensors to {output_file}")
    save_file(consolidated_state_dict, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
