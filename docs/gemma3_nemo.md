Gemma 3 Models
Gemma 3 introduces powerful and efficient open models in 1B/4B/12B/27B sizes, available as both large language models (LLMs) and vision-language models (VLMs). It is optimized for deployment across cloud, edge, and mobile devices. It builds on the transformer decoder architecture with improvements like grouped-query attention, advanced positional embeddings, and training techniques aligned with the Gemini family. More details are available in Google’s official release.

Gemma3 1B: A 1B parameter text-only model.

Gemma3 4B/12B/27B: A 4B/12B/27B parameter model with vision encoder.

Resources:

Hugging Face Gemma Collection: HF Gemma collection

Google Gemma Source Code: GitHub Repository

Import from Hugging Face to NeMo 2.0
To import the Hugging Face (HF) model and convert it to NeMo 2.0 format, run the following command. This step only needs to be performed once:

To convert Gemma 1B (LLM Only) model, or if you only want to convert the LLM component of the 4B/12B/27B model:

from nemo.collections import llm

if __name__ == '__main__':
    # Specify the Hugging Face model ID (e.g., Gemma3 1B Instruct Model)
    hf_model_id = 'google/gemma-3-1b-it'
    # Import the model and convert to NeMo 2.0 format
    llm.import_ckpt(
        model=llm.Gemma3Model(config=llm.Gemma3Config1B()),
        source=f"hf://{hf_model_id}",
    )
To convert Gemma 4B/12B/27B VLM model:

from nemo.collections import llm, vlm

if __name__ == '__main__':
    # Specify the Hugging Face model ID (e.g., Gemma3 4B Instruct Model)
    hf_model_id = 'google/gemma-3-4b-it'
    # Import the model and convert to NeMo 2.0 format
    llm.import_ckpt(
        model=vlm.Gemma3VLModel(config=vlm.Gemma3VLConfig4B()),
        source=f"hf://{hf_model_id}",
    )
The command above saves the converted file in the NeMo cache folder, located at: ~/.cache/nemo.

If needed, you can change the default cache directory by setting the NEMO_CACHE_DIR environment variable before running the script.

NeMo 2.0 Gemma 3 Scripts
The scripts for working with Gemma 3 models within the NeMo Framework are located in scripts/vlm/ and scripts/llm/ .

gemma3vl_generate.py: Performs inference (generation) using a fine-tuned or pre-converted Gemma 3 NeMo 2.0 VLM model.

Usage:

python scripts/vlm/gemma3vl_generate.py \
    --local_model_path=<path_to_nemo_model>
gemma3_generate.py: Performs inference (generation) using a fine-tuned or pre-converted Gemma 3 NeMo 2.0 LLM 1B model.

Usage:

python scripts/llm/gemma3_generate.py
Multi-Node Usage (Example with SLURM and Pyxis):

The following example demonstrates how to run text generation inference on 4 nodes with 8 GPUs each (total 32 GPUs) using SLURM. It assumes a containerized environment managed by Pyxis.

srun --mpi=pmix --no-kill \
    --container-image <path_to_container_image> \
    --container-mounts <necessary_mounts> \
    -N 4 --ntasks-per-node=8 -p <partition_name> --pty \
    bash -c " \
        python scripts/vlm/gemma3vl_generate.py \
            --local_model_path=<path_to_nemo_model> \
            --tp 8 \
            --pp 4 \
    "
gemma3vl_finetune.py: Fine-tunes a Gemma 3 4B model on a given dataset.

Usage:

torchrun --nproc_per_node=2 scripts/vlm/gemma3vl_finetune.py
gemma3_pretrain.py: Pretrains a Gemma 3 1B LLM model from scratch.

Usage:

torchrun scripts/llm/gemma3_pretrain.py
NeMo 2.0 Fine-Tuning Recipes
We provide pre-defined recipes for fine-tuning Gemma 3 VLM models (Gemma3VLModel) using NeMo 2.0 and NeMo-Run. These recipes configure a run.Partial for one of the nemo.collections.llm api functions introduced in NeMo 2.0. The recipes use the Gemma3VLMockDataModule for the data argument by default. The recipes are hosted in gemma3vl_4b, gemma3vl_12b, and gemma3vl_27b files.

The Gemma 3 1B LLM model recipe is hosted in gemma3_1b.

Note

The recipes use the Gemma3VLMockDataModule for the data argument. You are expected to replace the Gemma3VLMockDataModule with your custom dataset module.

By default, the instruct version of the model is loaded. To load a different model, set resume_path args in the recipe

We provide an example below on how to invoke the default recipe and override the data argument:

from nemo.collections import vlm, llm

# Get the fine-tuning recipe function (adjust for the specific Gemma VLM model)
finetune = vlm.gemma3vl_4b.finetune_recipe(
    name="gemma3_vl_4b_finetune",
    dir=f"/path/to/checkpoints",
    num_nodes=1,
    num_gpus_per_node=8,
    peft_scheme='lora', # or 'none' for full fine-tuning
)

# Finetune LLM recipe for Gemma
finetune = llm.gemma3_1b.finetune_recipe(
    name="gemma3_llm_1b_finetune",
    dir=f"/path/to/checkpoints",
    num_nodes=1,
    num_gpus_per_node=8,
    peft_scheme='lora', # or 'none' for full fine-tuning
)
By default, the fine-tuning recipe applies LoRA to all linear layers in the language model, including cross-attention layers, while keeping the vision model unfrozen.

To configure which layers to apply LoRA: Set finetune.peft.target_modules. For example, to apply LoRA only on the self-attention qkv projection layers, set finetune.peft.target_modules=["*.language_model.*.linear_qkv"].

To freeze the vision model: Set finetune.peft.freeze_vision_model=True.

To fine-tune the entire model without LoRA: Set peft_scheme='none' in the recipe argument.

Note

The configuration in the recipes is done using the NeMo-Run run.Config and run.Partial configuration objects. Please review the NeMo-Run documentation to learn more about its configuration and execution system.

Once you have your final configuration ready, you can execute it on any of the NeMo-Run supported executors. The simplest is the local executor, which just runs the pretraining locally in a separate process. You can use it as follows:

import nemo_run as run

run.run(finetune, executor=run.LocalExecutor())
Additionally, you can also run it directly in the same Python process as follows:

run.run(finetune, direct=True)
Bring Your Own Data
Replace the Gemma3VLMockDataModule in default recipes with your custom dataset module. Please refer to the Data Preparation to Use Megatron-Energon Dataloader for how to prepare your llava-like data for fine-tuning.

from nemo.collections.vlm as vlm

# Import your custom Gemma 3 data module and necessary configs
from nemo.collections.vlm.data.data_module import EnergonDataModule
from nemo.collections.vlm.gemma3vl.data.task_encoder import TaskEncoder as Gemma3VLTaskEncoder
from nemo.collections.vlm.gemma3vl.data.task_encoder import TaskEncoderConfig as Gemma3VLTaskEncoderConfig

# Define the fine-tuning recipe using the appropriate Gemma 3 recipe (adjust name if needed)
finetune = vlm.recipes.gemma3vl_4b.finetune_recipe(
    name="gemma3vl_4b_finetune",
    dir=f"/path/to/checkpoints",
    num_nodes=1,
    num_gpus_per_node=8,
    peft_scheme='lora', # or 'none'
)

# Example custom dataset configuration (replace with your actual data setup)
# Gemma 3 VL specific data configuration might be required here
task_encoder = Gemma3VLTaskEncoder(
    config=Gemma3VLTaskEncoderConfig(
        hf_path='google/gemma-3-4b-it', # Use the appropriate model path
    )
)
custom_data = EnergonDataModule(
    path="/path/to/energon/dataset", # Path to your Energon dataset
    train_encoder=task_encoder,
    seq_length=8192, # Adjust as needed
    global_batch_size=16, # Adjust based on GPU memory
    micro_batch_size=1, # Adjust based on GPU memory
    num_workers=8, # Adjust based on system capabilities
)

# Assign custom data to the fine-tuning recipe
finetune.data = custom_data
A comprehensive list of recipes that we currently support or plan to support soon is provided below for reference:

Recipe

Status

Gemma 3 VLM 4B Pretrain/LoRA/Full Fine-tuning

Yes

Gemma 3 VLM 12B Pretrain/LoRA/Full Fine-tuning

Yes

Gemma 3 VLM 27B Pretrain/LoRA/Full Fine-tuning

Yes

Gemma 3 LLM 1B Pretrain/LoRA/Full Fine-tuning

Yes