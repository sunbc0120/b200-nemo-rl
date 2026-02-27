**Title:** `save_consolidated=True` produces corrupted FSDP Safetensors natively (AssertionError in vLLM due to un-concatenated TP arrays)

**Describe the bug**
When utilizing PyTorch FSDP2 in NeMo-RL with `model_save_format: "safetensors"` and `save_consolidated: true` (as introduced in PR #1023), the framework fails to mathematically concatenate the sharded arrays along their partitioned dimensions before writing them to disk. 

The resulting `.safetensors` files are structurally corrupted. When vLLM or standard `transformers` APIs attempt to load them, they crash with an `AssertionError` regarding mismatched embedding tensors (e.g., loaded weight shape `[32768, 2048]` vs expected vocabulary `org_vocab_size` `262144`).

**Steps/Code to reproduce bug**
1. Configure a 1B/3B model GrPO training job utilizing FSDP2/TP1 on multiple GPUs.
2. Under `checkpointing`, configure:
    ```yaml
    model_save_format: "safetensors"
    save_consolidated: true
    ```
3. Allow the job to `save_model()`.
4. Attempt to load the `consolidated` output folder natively using vLLM (`vllm.LLM(model="...")`) or standard HuggingFace `AutoModelForCausalLM.from_pretrained(...)`.
5. Observe the `AssertionError` during the `VocabParallelEmbedding` shape check.

**Expected behavior**
The `_consolidate_safetensors_files` function inside the `nemo_automodel` submodule should iterate through the raw PyTorch DCP sharded `.safetensors` chunks and explicitly call `torch.cat(tensors, dim=X)` along the partitioned axes (e.g., `dim=0` for `embed_tokens` and `down_proj`, `dim=1` for `o_proj` depending on the row/column parallelism) to physically flatten the tensors into a unified shape *before* saving them to the final `model-00001-of-00001.safetensors` file.

**Additional context**
We analyzed the PR #1023 patch (`Implement automodel_checkpoint adapter`) and confirmed the NeMo-RL repository wrapper forwards the configuration correctly. However, the true bug resides upstream inside the `nemo_automodel` submodule's `_consolidate_safetensors_files` implementation. It simply streams the raw, fragmented chunks into new HuggingFace safetensor files and builds a `.hf_metadata` index claiming these un-merged shards represent the final model.

**Workaround Configured:**
We fell back to `model_save_format: "pytorch"` and `save_consolidated: false` and authored a brief python script (`manual_merge_fsdp.py`) that loads all the `.safetensors` shards into CPU RAM sequentially, detects partitioned tensors using `torch.equal()` checks across shards, and invokes `torch.cat()` to flatten the arrays. The resulting offline-merged safetensors file loads perfectly in vLLM and successfully achieves a 45.8% Pass@1 on MATH-500.
