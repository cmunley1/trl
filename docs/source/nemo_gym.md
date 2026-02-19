# NeMo Gym Integration

NVIDIA NeMo Gym is a library for building RL environments for large language models. This integration enables training models in NeMo Gym environments using TRL's GRPOTrainer with vLLM server mode.

The integration supports multi-step and multi-turn rollouts, multi-environment training, and any NeMo Gym environment.

## Before You Start

Complete these one-time setup steps before running training.

### Install TRL and NeMo Gym

```bash
cd trl/
uv venv
source .venv/bin/activate
uv sync --extra vllm
uv pip install fastapi uvicorn accelerate deepspeed wandb omegaconf

cd ..
git clone https://github.com/NVIDIA-NeMo/Gym
uv pip install -e Gym/
```

### Prepare a Dataset

In this example we will train a model on the workplace assistant environment, a multi-step tool use environment for common office scenarios. The dataset is available on Hugging Face. Use `ng_prepare_data` to download and prepare it:

```bash
cd Gym

ng_prepare_data \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

This creates `train.jsonl` and `validation.jsonl` in `data/workplace_assistant/`.

## Interactive Training

### Set Up

Update `examples/scripts/nemo_gym/config.yaml` to point to the dataset:

### Run Training

Training with NeMo Gym and TRL requires the vLLM server mode in TRL, therefore, a minimum of 2 GPUs is required:

1. **Start TRL vLLM Server on GPU 0**

   ```bash
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
     --model Qwen/Qwen3-4B-Instruct-2507 \
     --max-model-len 16384 \
     --host 0.0.0.0 \
     --port 8000
   ```

1. **Run Training on GPU 1**

   ```bash
   cd trl/examples/scripts/nemo_gym

   CUDA_VISIBLE_DEVICES=1 python grpo_nemo_gym.py
   ```

You should see training progress with completions logged to the terminal! Set up WandB to monitor training with in-depth.

## Multi-Node Training with Slurm

An example Slurm submission script is provided in `submit.sh`. Update it with your account, partition, container image, and paths, then submit with `sbatch submit.sh`.

## Multi-Environment Training

To train on multiple environments simultaneously, combine their datasets into a single JSONL file and add each environment's config to the `gym_configs` list in your training config. The training script automatically routes each example to the correct agent server based on its `agent_ref` field.

## Resources

- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym Documentation](https://docs.nvidia.com/nemo/gym/latest/)
- [Training Script](https://github.com/huggingface/trl/blob/main/examples/scripts/nemo_gym/grpo_nemo_gym.py)
- [TRL GRPO Trainer](grpo_trainer)
