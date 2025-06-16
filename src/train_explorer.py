import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import json

from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, List, NamedTuple, Optional
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import is_peft_model, send_to_device
from accelerate.utils.random import set_seed
from cosmos_tokenizer.image_lib import ImageTokenizer
from exploration_reward import (
    RewardFlags, compute_exploration_rewards_by_abs_value_with_ocr_v7,
    parse_reward_function_string)
from FlagEmbedding import BGEM3FlagModel
from gymnasium.vector.utils import concatenate as gym_concatenate
from gymnasium.vector.utils import create_empty_array
from world_model import LlamaForWorldModel, LlamaForWorldModelConfig
from paddlex import create_pipeline
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from rollout_buffer_async import RolloutBuffer
from schema import ActionSelection, jinja2_env
from screen_env import (AsyncState, FormatError,
                        SceenAgentEnv, create_empty_action,
                        generate_action_string, varify_generated_text)
from tqdm import tqdm
from trl.models import create_reference_model
from utils import LossTracker, sync_pytorch_to_vllm, tokenize_image
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from transformers import (AutoProcessor, AutoTokenizer, GenerationConfig,
                          HfArgumentParser, Qwen2_5_VLForConditionalGeneration)


@dataclass
class ScriptArguments:
    experiment_name: str = field(default="ScreenExplorer")

    # screen env
    video_width: int = field(default=1920)
    video_height: int = field(default=1080)
    vnc_ip: str = field(default="localhost")
    docker_tcp_url: str = field(default="tcp://127.0.0.1:2375")
    image_name: str = field(default="sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0")
    wait_after_action: float = field(default=1.0)
    mouse_coordinates_type: str = field(default="discrete")
    max_steps: Optional[int] = field(default=10, metadata={"help": "the maximum number of steps in an episode"})
    num_envs: int = field(default=8, metadata={"help": "the size of the group rollout buffer"})
    reset_env_interval: int = field(default=1, metadata={"help": "the episode interval for resetting the env"})

    cosmos_tokenizer_pretrained_ckpts: str = field(
        default="./pretrained_ckpts",
        metadata={"help": "the path to the pretrained ckpts of cosmos tokenizer"}
    )
    cosmos_tokenizer_model_name: Optional[str] = field(
        default="Cosmos-Tokenizer-CI16x16",
        metadata={"help": "the cosmos tokenizer model name, Cosmos-Tokenizer-CI8x8 or Cosmos-Tokenizer-CI16x16"}
    )
    text_embedding_model_device: str = field(default="cuda:0")

    use_world_model: bool = field(default=True, metadata={"help": "use world model or not"})
    world_model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B")
    world_model_device: str = field(default="cuda:0")
    load_world_model_weights: str = field(default="", metadata={"help": "Path to world model weights"})
    load_world_model_optimizer_weights: str = field(default="", metadata={"help": "Path to world model optimizer weights"})

    actor_model_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    load_lora_weights: str = field(default="", metadata={"help": "Path to LoRA weights"})
    load_actor_optimizer_weights: str = field(default="", metadata={"help": "Path to actor optimizer weights"})
    actor_model_device: str = field(default="cuda:0")
    actor_vllm_model_device: str = field(default="cuda:0")
    vllm_gpu_memory_utilization: float = field(default=0.2, metadata={"help": "vllm gpu memory utilization"})

    update_actor_delay: Optional[int] = field(default=5, metadata={"help": "update actor delay"})
    use_lora: Optional[bool] = field(default=True, metadata={"help": "use lora or not"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "lora alpha"})
    reward_components: str = field(default="all", metadata={"help": 'Reward components to enable/disable. Format: "component1,no_component2"'})

    # group_rollout_buffer
    max_completion_length: int = field(default=128)

    # training
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "the mixed precision mode"})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "the gradient accumulation steps"})
    max_episodes: Optional[int] = field(default=1000, metadata={"help": "the number of training episodes"})
    world_model_train_epoch: Optional[int] = field(default=3, metadata={"help": "world model train epoch"})
    world_model_train_batch_size: Optional[int] = field(default=2, metadata={"help": "world model train batch size"})
    world_model_lr: Optional[float] = field(default=4e-5, metadata={"help": "world model learning rate"})

    actor_model_train_batch_size: Optional[int] = field(default=1, metadata={"help": "actor model train batch size"})
    actor_model_lr: Optional[float] = field(default=4e-5, metadata={"help": "actor model learning rate"})

    world_model_max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for world model (0 to disable)"}
    )

    actor_model_max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for actor model (0 to disable)"}
    )

    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    epsilon_low: float = field(default=0.2, metadata={"help": "epsilon low"})
    epsilon_high: float = field(default=0.28, metadata={"help": "epsilon high"})

    eval_episode_interval: Optional[int] = field(default=10, metadata={"help": "frequency for evaluation and saving"})
    save_checkpoint_interval: Optional[int] = field(default=10, metadata={"help": "frequency for saving checkpoint"})

    seed: int = field(default=42)
    debug: Optional[bool] = field(default=False, metadata={"help": "debug mode"})


class ActionSelectionResult(NamedTuple):
    action: List[Any]
    action_obj: List[ActionSelection]
    action_string: List[str]
    format_reward: List[bool]
    prompt_ids: List[List[int]]
    completion_ids: List[List[int]]
    generated_text: List[str]


def get_vllm_imputs(args, processor, prompt, image):

    messages = [
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": args.video_height,
                        "resized_width": args.video_width
                    },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    mm_data = {
        "image": image
    }
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    return llm_inputs

def actor_select_action(
    args,
    envs,
    action_selection_prompt,
    obs_batch,
    actor_model_processor,
    actor_model_vllm_model,
    sampling_params,
    varify_generated_text_func,
):

    num_envs = envs.num_envs
    images = [Image.fromarray(obs, mode="RGB") for obs in obs_batch]

    batch_input = []
    for i in range(num_envs):
        llm_inputs = get_vllm_imputs(args, actor_model_processor, action_selection_prompt, images[i])
        batch_input.append(llm_inputs)

    outputs: List[RequestOutput] = actor_model_vllm_model.generate(batch_input, sampling_params=sampling_params)
    prompt_ids = [list(item.prompt_token_ids) for item in outputs]
    completion_ids = [list(item.outputs[0].token_ids) for item in outputs]
    generated_text = [item.outputs[0].text for item in outputs]

    batch_action = []
    batch_action_obj = []
    batch_format_reward = []
    batch_action_string = []
    for i in range(num_envs):
        action, action_obj, format_reward, action_string = varify_generated_text_func(generated_text[i])
        batch_action.append(action)
        batch_action_obj.append(action_obj)
        batch_format_reward.append(format_reward)
        batch_action_string.append(action_string)

    return ActionSelectionResult(
        action=batch_action,
        action_obj=batch_action_obj,
        action_string=batch_action_string,
        format_reward=batch_format_reward,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        generated_text=generated_text,
    )


def compute_group_advantages(rewards):
    """Compute advantages for all samples in buffer"""
    # Convert rewards to normalized advantages
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards)

    mean_reward = rewards.mean()
    std_reward = rewards.std() if rewards.std() > 0 else 1.0
    # Normalize rewards to create advantages
    advantages = ((rewards - mean_reward) / std_reward)
    return advantages

def actor_collect_rollouts(
    args,
    episode,
    rollout_buffer: RolloutBuffer,
    envs,
    actor_model_processor,
    actor_model_vllm_model,
    action_selection_prompt,
    sampling_params,
    reset_env_before_next_episode,
    max_steps=10,
    varify_generated_text_func=None,
    generate_action_string_func=None
):

    if envs._state == AsyncState.WAITING_CALL:
        envs.call_wait()

    num_envs = envs.num_envs
    all_intent_len = []
    terminated_by_error = False

    with torch.no_grad():
        for step in range(max_steps):
            try:
                obs_batch = envs.render()  # list of [ obs: [H, W, 3] ]
                actor_select_outputs = actor_select_action(
                    args=args,
                    envs=envs,
                    action_selection_prompt=action_selection_prompt,
                    obs_batch=obs_batch,
                    actor_model_processor=actor_model_processor,
                    actor_model_vllm_model=actor_model_vllm_model,
                    sampling_params=sampling_params,
                    varify_generated_text_func=varify_generated_text_func,
                )
            except Exception as e:
                terminated_by_error = True
                break

            group_action = actor_select_outputs.action
            group_action_obj = actor_select_outputs.action_obj
            group_action_string = actor_select_outputs.action_string
            group_format_reward = actor_select_outputs.format_reward
            group_generated_text = actor_select_outputs.generated_text

            record_group_action_obj = []
            select_action = group_action.copy()
            group_intent_string: List[ActionSelection] = []
            for i in range(num_envs):
                if group_format_reward[i] == False:
                    empty_action = create_empty_action()
                    select_action[i] = empty_action
                    group_action_string[i] = generate_action_string_func(empty_action)
                    record_group_action_obj.append(None)
                    group_intent_string.append(None)
                else:
                    record_group_action_obj.append(group_action_obj[i].model_dump())
                    group_intent_string.append(group_action_obj[i].intent)
                    all_intent_len.append(group_action_obj[i].intent_len)

            empty_array = create_empty_array(envs.single_action_space, n=num_envs)
            select_action = gym_concatenate(envs.single_action_space, select_action, out=empty_array)
            new_obs, task_reward, terminated, _, info = envs.step(select_action)  # new_obs: [H, W, 3]
            before_screen_ocr_results = []
            after_screen_ocr_results = []
            for i in range(num_envs):
                before_screen_ocr_results.append(json.loads(info["before_screen_ocr_results"][i]))
                after_screen_ocr_results.append(json.loads(info["after_screen_ocr_results"][i]))

            rollout_buffer.add_by_actor(
                episode_id=episode,
                env_id=list(range(num_envs)),
                step=step,
                batch_size=num_envs,
                prompt_ids=actor_select_outputs.prompt_ids,
                completion_ids=actor_select_outputs.completion_ids,
                format_rewards=actor_select_outputs.format_reward,  # list
                before_pixel_values=obs_batch,
                after_pixel_values=new_obs,
                generated_text=group_generated_text,
                action_dict=group_action,
                action_obj=record_group_action_obj,
                action_string=group_action_string,
                intent_string=group_intent_string,
                before_screen_ocr_results=before_screen_ocr_results,
                after_screen_ocr_results=after_screen_ocr_results,
            )

            if any(terminated):
                print(f"Episode {episode} terminated at step {step}.")
                terminated_by_error = True
                break

    if reset_env_before_next_episode:
        envs.call_async("reset")

    return {"intent_len": np.mean(all_intent_len)}, terminated_by_error


def extract_text_from_ocr_results(ocr_results):
    texts = []
    for one_image_ocr_result in ocr_results:
        all_text_on_screen = "Screen OCR: " + " ".join([item["text"] for item in one_image_ocr_result])
        texts.append(all_text_on_screen)
    return texts


def preparing_features(
    args,
    rollout_buffer: RolloutBuffer,
    text_embedding_func,
    tokenize_image_func,
):
    for indices, batch in tqdm(
        rollout_buffer.get_batch_for_preparing_features(batch_size=20),
        desc="Preparing Features", disable=not args.debug
    ):
        before_text_embeddings = text_embedding_func(extract_text_from_ocr_results(batch.before_screen_ocr_results))
        after_text_embeddings = text_embedding_func(extract_text_from_ocr_results(batch.after_screen_ocr_results))
        intent_string_embeddings = text_embedding_func(batch.intent_string)
        before_image_tokens = tokenize_image_func(batch.before_pixel_values)
        after_image_tokens = tokenize_image_func(batch.after_pixel_values)
        rollout_buffer.update_prepared_samples_features(
            indices,
            before_image_tokens=before_image_tokens,
            after_image_tokens=after_image_tokens,
            before_text_embeddings=before_text_embeddings,
            after_text_embeddings=after_text_embeddings,
            intent_string_embeddings=intent_string_embeddings
        )


def preparing_batch_for_world_model(
    rollout_buffer: RolloutBuffer,
    world_model_tokenizer
):
    for indices, batch_for_preparing_world_model in rollout_buffer.get_batch_for_preparing_world_model(batch_size=80):
        encoded_inputs = world_model_tokenizer.batch_encode_plus(
            batch_for_preparing_world_model.action_string,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
        )
        world_model_input_ids = encoded_inputs.input_ids
        world_model_attention_mask = encoded_inputs.attention_mask

        rollout_buffer.update_prepared_world_model_inputs(
            indices,
            world_model_input_ids=world_model_input_ids,
            world_model_attention_mask=world_model_attention_mask,
        )


def get_per_token_logps(
    model,
    input_ids,
    attention_mask,
    completion_mask,
    pixel_values,
    image_grid_thw,
):
    # Get model outputs
    logits = model.forward(
        input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    ).logits  # (B, L, V)

    # Shift logits and input_ids to align predictions with targets
    shifted_logits = logits[:, :-1, :]  # (B, L-1, V) - remove last position
    shifted_input_ids = input_ids[:, 1:]  # (B, L-1) - remove first position

    # Adjust the completion_mask mask to match the shifted dimensions
    # We need to remove the last position from the mask as well
    shifted_completion_mask = completion_mask[:, 1:]  # (B, L-1)

    # Compute log probabilities for all positions
    log_probs = shifted_logits.log_softmax(dim=-1)  # (B, L-1, V)

    # Gather log probs for the actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    # Apply the mask: set log probs to 0 for positions we want to ignore
    # This ensures they won't contribute to the loss
    masked_token_log_probs = token_log_probs * shifted_completion_mask.float()

    # Calculate entropy across vocabulary only for completion tokens (memory-efficient)
    with torch.no_grad():
        # Find the indices of tokens that are part of the completion
        batch_indices, seq_indices = torch.where(shifted_completion_mask)

        if len(batch_indices) > 0:  # Only compute if we have completion tokens
            # Extract only the logits for the completion tokens
            completion_logits = shifted_logits[batch_indices, seq_indices, :]  # (num_completion_tokens, V)

            # Compute softmax only for these tokens
            completion_probs = torch.softmax(completion_logits, dim=-1)  # (num_completion_tokens, V)

            # Compute entropy only for these tokens
            token_entropies = -torch.sum(completion_probs * torch.log(completion_probs + 1e-10), dim=-1)  # (num_completion_tokens)

            # Average the entropy values
            mean_entropy = token_entropies.mean()
        else:
            # No completion tokens found
            mean_entropy = torch.tensor(0.0, device=shifted_logits.device)

    # Return the per-token log probabilities
    return masked_token_log_probs, mean_entropy


def compute_loss(
    loss_type, model, inputs, ref_per_token_logps, beta=0.04, epsilon_low=0.2, epsilon_high=0.28, max_completion_length=None
):

    input_ids = inputs["action_model_input_ids"]
    attention_mask = inputs["action_model_attention_mask"]
    completion_mask = inputs["action_model_completion_mask"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    advantages = inputs["advantages"]

    per_token_logps, mean_entropy = get_per_token_logps(
        model,
        input_ids,
        attention_mask,
        completion_mask,
        pixel_values,
        image_grid_thw,
    )

    # Compute the KL divergence between the model and the reference model
    completion_mask = completion_mask[:, 1:]  # (B, L-1)
    per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
    completion_token_num = completion_mask.sum(dim=1).clamp(min=1.0)

    # huggingface trl grpo loss
    coef_1 = torch.exp(per_token_logps - per_token_logps.detach())  # discussions https://github.com/huggingface/trl/pull/2565#issuecomment-2673805226
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = - torch.min(per_token_loss1, per_token_loss2)

    if beta != 0.0:
        per_token_loss = per_token_loss + beta * per_token_kl

    loss = (
        (per_token_loss * completion_mask).sum(-1) / completion_token_num
    ).mean()

    mean_kl = (
        (per_token_kl * completion_mask).sum(dim=1) / completion_token_num
    ).mean().detach()

    return {
        "loss": loss,
        "mean_kl": mean_kl,
        "mean_entropy": mean_entropy
    }


def main():
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    experiment_name = args.experiment_name
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name += f"_{datetime_str}" + uuid4().hex[:6]

    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        experiment_name += "_SLURM_" + slurm_job_id
    if args.debug:
        experiment_name += "_debug"
    experiment_save_dir = f"logs/{experiment_name}"
    os.makedirs(experiment_save_dir, exist_ok=True)

    wandb.init(
        project="ScreenExplorer",
        name=experiment_name,
        config=args,
    )

    set_seed(args.seed)

    # Reward Components
    reward_flags: RewardFlags = parse_reward_function_string(args.reward_components)
    assert not (not args.use_world_model and reward_flags.world_model), "world_model reward flag should be False when not using world model"

    args.use_image = reward_flags.image
    args.use_text = reward_flags.text

    pretrained_ckpts_dir = args.cosmos_tokenizer_pretrained_ckpts
    image_tokenizer = ImageTokenizer(
        checkpoint_enc=f'{pretrained_ckpts_dir}/{args.cosmos_tokenizer_model_name}/encoder.jit',
        checkpoint_dec=f'{pretrained_ckpts_dir}/{args.cosmos_tokenizer_model_name}/decoder.jit',
    )
    tokenize_image_func = partial(tokenize_image, image_tokenizer)
    start_episode = 0

    if args.use_lora:
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            r=args.lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        )
    else:
        lora_config = None

    actor_model_vllm_model = LLM(
        model=args.actor_model_path,
        device=args.actor_vllm_model_device,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        max_model_len=5000 + args.max_completion_length
    )
    if args.use_world_model:
        world_model_device = torch.device(args.world_model_device)
        world_model_tokenizer = AutoTokenizer.from_pretrained(args.world_model_name_or_path)
        world_model_tokenizer.pad_token = world_model_tokenizer.eos_token

        world_config = LlamaForWorldModelConfig.from_pretrained(
            args.world_model_name_or_path,
            use_image=reward_flags.image,
            use_text=reward_flags.text
        )

        if args.load_world_model_weights:
            print(f"Loading world model weights from {args.load_world_model_weights}")
            world_model = LlamaForWorldModel.from_pretrained(
                args.load_world_model_weights,
                config=world_config,
                device_map={"": world_model_device.index},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )  
        else:
            print(f"Loading world model weights from {args.world_model_name_or_path}")
            world_model = LlamaForWorldModel.from_pretrained(
                args.world_model_name_or_path,
                config=world_config,
                device_map={"": world_model_device.index},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        world_model_optimizer = torch.optim.AdamW(
            [p for p in world_model.parameters() if p.requires_grad],
            lr=args.world_model_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        if args.load_world_model_optimizer_weights:
            print(f"Loading world model optimizer weights from {args.load_world_model_optimizer_weights}")
            checkpoint = torch.load(args.load_world_model_optimizer_weights)
            world_model_optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"World model optimizer weights loaded successfully (from episode {checkpoint['epoch']})")
        
        world_model_accelerator = Accelerator(mixed_precision=args.mixed_precision)
        world_model, world_model_optimizer = world_model_accelerator.prepare(world_model, world_model_optimizer, device_placement = [False, True])

    actor_model_device = torch.device(args.actor_model_device)
    actor_model_processor = AutoProcessor.from_pretrained(args.actor_model_path)
    actor_model_tokenizer = actor_model_processor.tokenizer
    actor_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.actor_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": actor_model_device.index},
    )

    if args.use_lora:
        if args.load_lora_weights:
            print(f"Loading LoRA weights from {args.load_lora_weights}")
            actor_model = PeftModel.from_pretrained(actor_model, args.load_lora_weights, is_trainable=True)
            print("LoRA weights loaded successfully")
            sync_pytorch_to_vllm(actor_model, actor_model_vllm_model)
        else:
            actor_model = get_peft_model(actor_model, lora_config)
        ref_actor_model = None
    else:
        ref_actor_model = create_reference_model(actor_model)

    actor_model_optimizer = torch.optim.AdamW(
        [p for p in actor_model.parameters() if p.requires_grad],
        lr=args.actor_model_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    if args.load_actor_optimizer_weights:
        print(f"Loading actor optimizer weights from {args.load_actor_optimizer_weights}")
        checkpoint = torch.load(args.load_actor_optimizer_weights, map_location=actor_model_device)
        actor_model_optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['epoch'] + 1
        print(f"Actor optimizer weights loaded successfully (from episode {checkpoint['epoch']})")
              
    actor_model_accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    actor_model, actor_model_optimizer = actor_model_accelerator.prepare(actor_model, actor_model_optimizer, device_placement = [False, False])

    env_config = {
        "experiment_name": experiment_name,
        "image_name": args.image_name,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "max_steps": -1,  # no limit in env, but will limit it in actor_collect_rollouts
        "mouse_coordinates_type": args.mouse_coordinates_type,
        "vnc_ip": args.vnc_ip,
        "docker_tcp_url": args.docker_tcp_url,
        "wait_after_action": args.wait_after_action,
        "use_remote_clipboard": True
    }

    # Text Embedding Model
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=args.text_embedding_model_device)

    def text_embedding_func(texts):
        batch_embeddings = text_embedding_model.encode(texts, batch_size=32, max_length=8192)['dense_vecs']
        batch_embeddings = torch.tensor(batch_embeddings)
        return batch_embeddings
    print("Text embedding model loaded successfully.")
    sample_texts = ["Sample text 1", "Sample text 2"]
    text_embedding_func(sample_texts)

    # OCR Model
    ocr_pipeline = create_pipeline(pipeline="OCR", device="cpu", use_hpip=True)

    def ocr_func(images):
        predict_results = ocr_pipeline.predict(
            input=images,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        batch_texts = []
        ocr_results = []
        for ocr_result in predict_results:
            all_text_on_screen = "Screen OCR: " + " ".join(ocr_result["rec_texts"])
            batch_texts.append(all_text_on_screen)
            item_results = []
            for text, rec_boxes in zip(ocr_result["rec_texts"], ocr_result["rec_boxes"]):
                item_results.append({
                    "text": text,
                    "boxes": rec_boxes.tolist()
                })
            ocr_results.append(item_results)

        return ocr_results


    env_cls = SceenAgentEnv
    envs = gym.vector.AsyncVectorEnv([
        lambda env_rank=i: env_cls(env_config, render_mode="rgb_array", env_rank=env_rank, ocr_func=ocr_func)
        for i in range(args.num_envs)
    ])

    varify_generated_text_func = partial(
        varify_generated_text,
        mouse_coordinates_type=args.mouse_coordinates_type,
        video_width=args.video_width,
        video_height=args.video_height,
    )
    generate_action_string_func = partial(
        generate_action_string,
        mouse_coordinates_type=args.mouse_coordinates_type,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    action_selection_by_vlm_prompt = jinja2_env.get_template("action_selection_by_vlm_en.txt")
    action_selection_prompt = action_selection_by_vlm_prompt.render(
        video_height=args.video_height,
        video_width=args.video_width,
    )

    guided_decoding_params = GuidedDecodingParams(json=ActionSelection.model_json_schema())
    sampling_params = SamplingParams(
        temperature=1.0,
        repetition_penalty=1.05,
        max_tokens=args.max_completion_length,
        stop_token_ids=[actor_model_tokenizer.eos_token_id],
        guided_decoding=guided_decoding_params
    )

    envs.reset()
    for episode in range(start_episode, args.max_episodes):
        rollout_buffer = RolloutBuffer(
            actor_model_padding_token_id=actor_model_tokenizer.pad_token_id,
            world_model_padding_token_id=world_model_tokenizer.pad_token_id if args.use_world_model else None,
            use_world_model=args.use_world_model,
            use_image=reward_flags.image,
            use_text=reward_flags.text,
            device="cpu",
        )
        tracker = LossTracker()
        reset_env_before_next_episode = False
        if (episode + 1) % args.reset_env_interval == 0:
            reset_env_before_next_episode = True

        stats, terminated_by_error = actor_collect_rollouts(
            args=args,
            episode=episode,
            rollout_buffer=rollout_buffer,
            envs=envs,
            actor_model_processor=actor_model_processor,
            actor_model_vllm_model=actor_model_vllm_model,
            action_selection_prompt=action_selection_prompt,
            sampling_params=sampling_params,
            reset_env_before_next_episode=reset_env_before_next_episode,
            max_steps=args.max_steps,
            varify_generated_text_func=varify_generated_text_func,
            generate_action_string_func=generate_action_string_func
        )
        if terminated_by_error:
            print(f"Episode {episode} terminated by error.")
            wandb.log({
                "terminated_by_error": True,
                "episode": episode,
            }, step=episode)
            continue

        preparing_features(args, rollout_buffer, text_embedding_func, tokenize_image_func)

        if args.use_world_model:
            # processing input_ids for world model
            preparing_batch_for_world_model(rollout_buffer, world_model_tokenizer)

            world_model.train()
            # Update the world model
            for epoch in range(args.world_model_train_epoch):
                for batch_indices, rollout_data in tqdm(
                    rollout_buffer.get_batch_for_world_model(batch_size=args.world_model_train_batch_size),
                    desc="Training World Model", disable=not args.debug
                ):
                    # with world_model_accelerator.accumulate(world_model):
                        with world_model_accelerator.autocast():
                            inputs = {
                                "input_ids": rollout_data.world_model_input_ids,
                                "attention_mask": rollout_data.world_model_attention_mask,
                                "input_image_tokens": rollout_data.before_image_tokens,
                                "label_image_tokens": rollout_data.after_image_tokens,
                                "input_text_embeddings": rollout_data.before_text_embeddings,
                                "label_text_embeddings": rollout_data.after_text_embeddings,
                            }
                            inputs = send_to_device(inputs, world_model_device)

                            world_model_returns = world_model(**inputs)
                            world_model_loss = world_model_returns.loss
                            world_model_accelerator.backward(world_model_loss)
                            if world_model_accelerator.sync_gradients:
                                world_model_accelerator.clip_grad_norm_(world_model.parameters(), args.world_model_max_grad_norm)
                            world_model_optimizer.step()
                            world_model_optimizer.zero_grad()

                            tracker.add_losses({
                                "world_model_loss": world_model_loss.item(),
                                **world_model_returns.losses
                            })
                            if epoch == args.world_model_train_epoch - 1:
                                rollout_buffer.update_world_model_output(
                                    batch_indices,
                                    pred_obs=world_model_returns.pred_obs,
                                    pred_text_embeddings=world_model_returns.pred_text_embeddings,
                                )

        # Compute rewards for actor model
        for batch_indices, rollout_data in tqdm(
            rollout_buffer.get_samples_for_reward_compute(),
            desc="Computing Advantages", disable=not args.debug
        ):
            reward_returns = compute_exploration_rewards_by_abs_value_with_ocr_v7(
                reward_flags=reward_flags,
                before_image_tokens=rollout_data.before_image_tokens,
                after_image_tokens=rollout_data.after_image_tokens,
                pred_after_image_tokens=rollout_data.pred_after_image_tokens if args.use_world_model and args.use_image else None,
                before_text_embeddings=rollout_data.before_text_embeddings,
                after_text_embeddings=rollout_data.after_text_embeddings,
                pred_text_embeddings=rollout_data.pred_text_embeddings if args.use_world_model and args.use_text else None,
                batch_action_dict=rollout_data.action_dict,
                before_screen_ocr_results=rollout_data.before_screen_ocr_results,
                intent_string_embeddings=rollout_data.intent_string_embeddings,
                method="cosine", text_embedding_func=text_embedding_func,
            )

            rewards = reward_returns["rewards"]
            rollout_buffer.update_reward(batch_indices, rewards=rewards)
            tracker.add_losses({k: v.mean().item() for k, v in reward_returns.items()})

        # Compute the advantages for actor model
        rewards = rollout_buffer.get_all_rewards()  # masked by format rewards
        advantages = compute_group_advantages(args.loss_type, rewards)
        rollout_buffer.update_advantages(advantages=advantages)

        # Update the actor model
        actor_model.train()
        if (episode > args.update_actor_delay and args.use_world_model) or args.debug:
            for rollout_data in tqdm(
                rollout_buffer.get_batch_for_actor(batch_size=args.actor_model_train_batch_size),  # actor保持online训练，通过所有样本。
                desc="Training Actor", disable=not args.debug
            ):
                image_inputs = actor_model_processor.image_processor(images=rollout_data.before_pixel_values)  # Qwen2VLImageProcessor
                pixel_values = torch.from_numpy(image_inputs["pixel_values"])
                image_grid_thw = torch.from_numpy(image_inputs["image_grid_thw"])
                # update_actor
                inputs = {
                    "action_model_input_ids": rollout_data.action_model_input_ids,
                    "action_model_attention_mask": rollout_data.action_model_attention_mask,
                    "action_model_completion_mask": rollout_data.action_model_completion_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "advantages": rollout_data.advantages,
                }
                inputs = send_to_device(inputs, actor_model_device)
                with torch.no_grad():
                    if ref_actor_model is None and is_peft_model(actor_model):
                        with actor_model.disable_adapter():
                            ref_per_token_logps, ref_entropy = get_per_token_logps(
                                actor_model,
                                inputs["action_model_input_ids"],
                                inputs["action_model_attention_mask"],
                                inputs["action_model_completion_mask"],
                                inputs["pixel_values"],
                                inputs["image_grid_thw"],
                            )

                    else:
                        ref_per_token_logps, ref_entropy = get_per_token_logps(
                            ref_actor_model,
                            inputs["action_model_input_ids"],
                            inputs["action_model_attention_mask"],
                            inputs["action_model_completion_mask"],
                            inputs["pixel_values"],
                            inputs["image_grid_thw"],
                        )
                    ref_per_token_logps = ref_per_token_logps.detach()

                with actor_model_accelerator.accumulate(actor_model):
                    with actor_model_accelerator.autocast():
                        loss_dict = compute_loss(
                            args.loss_type, actor_model, inputs, ref_per_token_logps, 
                            beta=args.beta,
                            epsilon_low=args.epsilon_low,
                            epsilon_high=args.epsilon_high,
                            max_completion_length=args.max_completion_length
                        )
                        actor_loss = loss_dict["loss"]
                        actor_model_accelerator.backward(actor_loss)
                        if actor_model_accelerator.sync_gradients:
                            actor_model_accelerator.clip_grad_norm_(actor_model.parameters(), args.actor_model_max_grad_norm)
                        actor_model_optimizer.step()
                        actor_model_optimizer.zero_grad()

                        tracker.add_losses({
                            "actor_loss": actor_loss.item(),
                            "mean_kl": loss_dict["mean_kl"].item(),
                            "mean_entropy": loss_dict["mean_entropy"].item(),
                            "ref_entropy": ref_entropy.item()
                        })


            # Sync the actor model to VLLM
            sync_pytorch_to_vllm(actor_model, actor_model_vllm_model)

        rollout_stats = rollout_buffer.get_group_stats(episode_id=episode)
        losses = tracker.get_average_losses()  # Get averaged losses for logging
        print(f"Episode {episode}, " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))
        print(f"Episode {episode}, " + ", ".join(f"{k}={v:.4f}" for k, v in rollout_stats.items()))
        wandb.log({
            "rollout_stats": rollout_stats,
            "train": losses,
            "episode": episode,
        }, step=episode)

        del rollout_buffer

        if episode % args.save_checkpoint_interval == 0 and episode > 0:
            checkpoint_dir = os.path.join(experiment_save_dir, f"episode_{episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            # save_checkpoint_and_optimizer_state
            actor_model.save_pretrained(os.path.join(checkpoint_dir, f"actor_model_{episode}"))

            if args.use_world_model:
                world_model.save_pretrained(os.path.join(checkpoint_dir, f"world_model_{episode}"))
                torch.save(
                    {
                        'optimizer': world_model_optimizer.state_dict(),
                        'epoch': episode,
                    },
                    os.path.join(checkpoint_dir, f"world_model_optimizer_{episode}.pt")
                )

            torch.save(
                {
                    'optimizer': actor_model_optimizer.state_dict(),
                    'epoch': episode,
                },
                os.path.join(checkpoint_dir, f"actor_optimizer_{episode}.pt")
            )

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    main()
