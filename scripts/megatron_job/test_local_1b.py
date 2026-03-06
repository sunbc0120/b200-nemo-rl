import os
import torch
import sys
sys.path.append("/opt/nemo-rl")
sys.path.append("/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM")
sys.path.append("/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src")
sys.path.append("/opt/nemo-rl/3rdparty/modelopt-workspace/modelopt/src")

# Remove Ray cluster logging noise completely
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Import Megatron Policy worker infrastructure
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker
from omegaconf import OmegaConf

import ray

def main():
    model_name = "google/gemma-3-1b-it"
    print(f"Testing simplified Megatron generation for {model_name}...")

    ray.init(ignore_reinit_error=True)

    # A minimal Megatron configuration for local 1 GPU testing
    config = OmegaConf.create({
        "model_name": model_name,
        "megatron_cfg": {
            "converter_type": "Gemma3ForCausalLM",
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": False,
            "apply_rope_fusion": False,
            "bias_activation_fusion": False, # Important for Gemma 3
            "untie_embeddings_and_output_weights": False,
            "moe_enable_deepep": False,
            "pipeline_dtype": "bfloat16",
            "optimizer": {
                "use_distributed_optimizer": False,
                "optimizer": "adam",
            },
            "distributed_data_parallel_config": {
                "overlap_param_gather": False,
                "overlap_grad_reduce": False, # Multimodal vision compatibility
            },
            "fp8_cfg": {"enabled": False},
        },
        "tokenizer": {
            "name": model_name,
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "generation": {
            "max_new_tokens": 128,
            "temperature": 1.0,
            "top_p": 1.0,
            "backend": "megatron",
            "mcore_generation_config": {
                "buffer_size_gb": 4, # Less RAM for 1B
                "block_size_tokens": 256,
                "use_cuda_graphs_for_non_decode_steps": False,
                "enable_chunked_prefill": True,
            }
        },
        "max_total_sequence_length": 4096,
        "dp_size": 1,
    })

    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Initializing Megatron Policy Worker...")
    try:
        worker = MegatronPolicyWorker.remote(
            config=config,
            megatron_cfg=config.megatron_cfg,
            max_total_sequence_length=config.max_total_sequence_length,
            dp_size=config.dp_size,
            tokenizer=hf_tokenizer,
        )
        print("Worker initialized successfully natively on Ray!")
    except Exception as e:
        print(f"Failed to initialize Worker: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Now let's try a test query
    print("Preparing test prompt...")
    prompts = ["<bos><start_of_turn>user\nWhat is 2+2?\n<end_of_turn>\n<start_of_turn>model\n"]
    
    # We must tokenize it exactly as the environment would
    print("Tokenizing prompt using Megatron's internal tokenization flow...")
    input_ids = hf_tokenizer(prompts, return_tensors="pt").input_ids.cuda()
    
    # The NeMo layer expects an array of Python strings, not tensors, when calling .generate()!
    print("Executing generation forward pass...")
    outputs = ray.get(worker.generate.remote(
        prompts=prompts,
        generation_cfg=config.generation,
    ))
    
    print("\n--- GENERATION COMPLETE ---")
    for i, out in enumerate(outputs):
        print(f"Sample {i}:")
        print(repr(out["text"]))
        print("-" * 30)

if __name__ == "__main__":
    main()
