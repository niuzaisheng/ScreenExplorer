<p align="center">
<h1 align="center"> ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World </h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.19095">
    <img src="https://img.shields.io/badge/arXiv-2505.19095-b31b1b.svg" alt="arXiv">
  </a>
</p>

We introduce ScreenExplorer, a VLM trained via Group Relative Policy Optimization(GRPO) in real, dynamic, and open-ended GUI environments for diverse exploration. ScreenExplorer is trained to explore and interact with the screen environment, learning to interact effectively with environments based on screenshots and a fixed instruction to encourage exploration.

## Demo Videos

**ScreenExplorer-3B-E1 Video**

https://github.com/user-attachments/assets/08c396e3-b027-4eaf-93ae-a54f69b410ce

**ScreenExplorer-7B-E1 Video**

https://github.com/user-attachments/assets/803b9dd7-9ac7-47ea-841b-54e221b5670a

## Project Structure

    ScreenExplorer/
    ├── requirements.txt
    └── src/
        ├── schema
        │   ├── action_selection_by_vlm_en.txt: The fixed instruction to encourage exploration
        │   ├── action_selection.py:            Action selection schema
        │   └── __init__.py
        ├── screen_env
        │   ├── asyncvnc.py:                    VNC client for screen interaction
        │   └── screen_env.py:                  Environment wrapper for screen-based interaction
        ├── train_explorer.py:                  Main training script for the explorer agent
        ├── exploration_reward.py:              Exploration rewards
        ├── online_eval.py:                     Online evaluation script
        ├── rollout_buffer.py:                  Manages experience rollouts for training
        ├── utils.py
        └── world_model.py:                     World model implementation

## Preparation

1. Download Cosmos-Tokenizer-CI16x16 pretrained checkpoint from [here](https://huggingface.co/collections/nvidia/cosmos-tokenizer-672b93023add81b66a8ff8e6) and put it in `src/pretrained_ckpts/` directory.

2. Make sure you have downloaded base model `Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-VL-yB-Instruct` from huggingface. And meta-llama/Llama-3.2-1B for world model. 

3. Setup docker environment for screen environment:

```bash
docker pull sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0 # in global
# or 
docker pull ccr.ccs.tencentyun.com/screenagent/screenagent:2.0 # in China
```

## Run Training

Train 3B model on 1 GPU:

```bash
cd src
export CUDA_VISIBLE_DEVICES=0
python train_explorer.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --world_model_name_or_path meta-llama/Llama-3.2-1B \
  --cosmos_tokenizer_pretrained_ckpts ./pretrained_ckpts \
  --cosmos_tokenizer_model_name Cosmos-Tokenizer-CI16x16 \
  --image_name sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0 \
  --save_checkpoint_interval 10
```

Train 7B model on 2 GPU:

```bash
cd src
export CUDA_VISIBLE_DEVICES=0,1
python train_explorer.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --world_model_name_or_path meta-llama/Llama-3.2-1B \
  --cosmos_tokenizer_pretrained_ckpts ./pretrained_ckpts \
  --cosmos_tokenizer_model_name Cosmos-Tokenizer-CI16x16 \
  --image_name sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0 \
  --actor_model_device "cuda:1" \
  --save_checkpoint_interval 10
```

## Run Online Evaluation:

Evaluate base 3B model on 1 GPU:

```bash
cd src
export CUDA_VISIBLE_DEVICES=0
python online_eval.py --eval_episodes 20 --model_type vllm --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --temperature 1.0
```

Evaluate checkpoint of 3B model on 1 GPU:

```bash
cd src
export CUDA_VISIBLE_DEVICES=0
python online_eval.py \
--eval_episodes 20 \
--model_type vllm \
--model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
--load_lora_weights logs/<path_to_your_experiment_checkpoint_dir>/episode_100/actor_model_100 \
--temperature 1.0
```
