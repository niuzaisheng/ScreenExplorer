import os
import torch
from PIL import Image
from datetime import datetime
import time
from functools import partial
from tqdm import tqdm
from typing import List, Optional

from transformers import HfArgumentParser
from accelerate.utils.random import set_seed

from cosmos_tokenizer.image_lib import ImageTokenizer
from FlagEmbedding import BGEM3FlagModel

from screen_env import SCEENAGENT_ENV_VERSION, SceenAgentEnv, varify_generated_text, parse_action, create_empty_action_v4, action_dict_covert_to_numpy, AsyncState,  FormatError, UnSupportedActionError, APIError
from schema import jinja2_env, ActionSelection
from gymnasium.vector.utils import concatenate as gym_concatenate
from dataclasses import dataclass, field

from utils import convert_image_to_base64, tokenize_image, ActionSelectionResult


@dataclass
class ScriptArguments:
    experiment_name: str = field(default="Online Eval")

    # Environment settings
    max_steps: int = field(default=10)
    eval_episodes: int = field(default=20)
    video_width: int = field(default=1920)
    video_height: int = field(default=1080)
    wait_after_action: float = field(default=1.0)
    mouse_coordinates_type: str = field(default="discrete")
    vnc_ip: str = field(default="localhost")
    docker_tcp_url: str = field(default="tcp://127.0.0.1:2375")
    image_name: str = field(default="sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0")

    cosmos_tokenizer_pretrained_ckpts: str = field(
        default="./pretrained_ckpts",
        metadata={"help": "the path to the pretrained ckpts of cosmos tokenizer"}
    )
    cosmos_tokenizer_model_name: Optional[str] = field(
        default="Cosmos-Tokenizer-CI16x16",
        metadata={"help": "the cosmos tokenizer model name, Cosmos-Tokenizer-CI8x8 or Cosmos-Tokenizer-CI16x16"}
    )
    text_embedding_model_device: str = field(default="cuda:0")
    actor_vllm_model_device: str = field(default="cuda:0")
    vllm_gpu_memory_utilization: float = field(default=0.2, metadata={"help": "vllm gpu memory utilization"})

    # Model settings
    model_type: str = field(default="vllm", metadata={"choices": ["vllm", "openai", "openai-computer-use", "doubao"]})
    model_name_or_path: str = field(default="")
    load_lora_weights: str = field(default="", metadata={"help": "Path to LoRA weights"})
    temperature: float = field(default=1.0)
    repetition_penalty: float = field(default=1.05)
    max_completion_length: int = field(default=128)
    # Output settings
    output_base_dir: str = field(default="eval_results")
    seed: int = field(default=42)

def convert_tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array"""
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.detach().cpu().numpy()
    return tensor


def main():

    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    print(args)

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.output_base_dir, exist_ok=True)
    model_name_or_path_replace = args.model_name_or_path.replace("/", "__")
    if args.load_lora_weights:
        model_name_or_path_replace += "_with_lora"
    experiment_name = f"{args.experiment_name}_{args.model_type}_{model_name_or_path_replace}_temp_{args.temperature}_{SCEENAGENT_ENV_VERSION}_{datetime_str}"
    output_dir = os.path.join(args.output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.load_lora_weights:
        with open(f"{output_dir}/lora_weights.txt", "w") as f:
            f.write(args.load_lora_weights)

    set_seed(args.seed)

    pretrained_ckpts_dir = args.cosmos_tokenizer_pretrained_ckpts
    image_tokenizer_device = torch.device("cuda:0")
    image_tokenizer = ImageTokenizer(checkpoint_enc=f'{pretrained_ckpts_dir}/{args.cosmos_tokenizer_model_name}/encoder.jit',
                                     checkpoint_dec=f'{pretrained_ckpts_dir}/{args.cosmos_tokenizer_model_name}/decoder.jit',
                                     device=image_tokenizer_device)
    tokenize_image_func = partial(tokenize_image, image_tokenizer)

    # Setup environment
    env_config = {
        "experiment_name": experiment_name,
        "image_name": args.image_name,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "max_steps": args.max_steps,
        "mouse_coordinates_type": args.mouse_coordinates_type,
        "vnc_ip": args.vnc_ip,
        "docker_tcp_url": args.docker_tcp_url,
        "wait_after_action": args.wait_after_action,
        "use_remote_clipboard": True
    }

    env = SceenAgentEnv(env_config, render_mode="rgb_array")

    varify_generated_text_func = partial(
        varify_generated_text,
        mouse_coordinates_type=args.mouse_coordinates_type,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    # Text Embedding Model
    text_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=args.text_embedding_model_device)

    def text_embedding_func(texts, return_tensor=False):
        batch_embeddings = text_embedding_model.encode(texts, batch_size=32, max_length=8192)['dense_vecs']
        if return_tensor:
            batch_embeddings = torch.tensor(batch_embeddings)
        return batch_embeddings

    print("Text embedding model loaded successfully.")
    sample_texts = ["Sample text 1", "Sample text 2"]
    text_embedding_func(sample_texts)

    if args.model_type == "vllm":
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams, RequestOutput
        from vllm.sampling_params import GuidedDecodingParams

        actor_model_processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        actor_model_tokenizer = actor_model_processor.tokenizer
        guided_decoding_params = GuidedDecodingParams(json=ActionSelection.model_json_schema())
        sampling_params = SamplingParams(
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_completion_length,
            stop_token_ids=[actor_model_tokenizer.eos_token_id],
            guided_decoding=guided_decoding_params
        )

        actor_vllm_model = LLM(
            model=args.model_name_or_path,
            device=args.actor_vllm_model_device,
            # limit_mm_per_prompt={"image": 1, "video": 0},
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            # enable_chunked_prefill=True,
            max_model_len=5000 + args.max_completion_length
        )

        if args.load_lora_weights:
            print(f"Loading LoRA weights from {args.load_lora_weights}")
            from peft import LoraConfig, PeftModel
            from transformers import Qwen2_5_VLForConditionalGeneration
            from utils import sync_pytorch_to_vllm

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": args.actor_vllm_model_device},
            )
            model_with_lora = PeftModel.from_pretrained(base_model, args.load_lora_weights)
            sync_pytorch_to_vllm(model_with_lora, actor_vllm_model)
            model_with_lora = None
            del base_model
            torch.cuda.empty_cache()
            print("LoRA weights loaded successfully")

        # Load prompt template
        action_selection_by_vlm_prompt = jinja2_env.get_template("action_selection_by_vlm_en.txt")
        action_selection_prompt = action_selection_by_vlm_prompt.render(
            video_height=args.video_height,
            video_width=args.video_width,
        )

        def get_vllm_imputs(prompt, image):

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

            prompt = actor_model_processor.apply_chat_template(
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

        def select_action(one_obs):
            image = Image.fromarray(one_obs, mode="RGB")
            llm_inputs = get_vllm_imputs(action_selection_prompt, image)
            outputs: List[RequestOutput] = actor_vllm_model.generate([llm_inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text
            print("Generated text:", generated_text)
            action, action_obj, format_reward, action_string = varify_generated_text_func(generated_text)
            return ActionSelectionResult(
                action=action,
                format_reward=format_reward,
                generated_text=generated_text,
                unsupported_action=False,
            )

    elif args.model_type == "doubao":
        from baselines.doubao_adapter import call_vlm, parse_doubao_action_output, doubao_to_screenagent_actions_v4
        user_prompt = "Your goal is to explore this environment as much as possible within a limited number of steps. Please select a meaningful action to continue exploring. Note that opening icons on the desktop requires a double click. You must only use mouse and keyboard inputs. No other tools or input devices are permitted."
        model = "doubao-1.5-ui-tars-250328"

        def select_action(one_obs):
            image = Image.fromarray(one_obs, mode="RGB")
            generated_text = call_vlm(image, user_prompt, model, temperature=args.temperature)
            doubao_style_action_dict = parse_doubao_action_output(generated_text, image.width, image.height)

            action = None
            format_reward = False
            unsupported_action = False
            try:
                action = doubao_to_screenagent_actions_v4(doubao_style_action_dict, image.width, image.height)
                format_reward = True
            except FormatError as e:
                format_reward = False
            except UnSupportedActionError as e:
                unsupported_action = True

            return ActionSelectionResult(
                action=action,
                format_reward=format_reward,
                generated_text=generated_text,
                unsupported_action=unsupported_action
            )

    elif args.model_type == "openai":
        import openai
        openai_client = openai.Client(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
        generate_kwargs = {
            "temperature": args.temperature,
            "max_tokens": args.max_completion_length,
            "top_p": 1,
            "frequency_penalty": args.repetition_penalty,
            "response_format": ActionSelection
        }
        action_selection_by_vlm_prompt = jinja2_env.get_template("action_selection_by_vlm_en.txt")

        def select_action(one_obs):
            image = Image.fromarray(one_obs, mode="RGB")
            image_base64 = convert_image_to_base64(image)
            prompt = action_selection_by_vlm_prompt.render(
                video_height=args.video_height,
                video_width=args.video_width,
            )

            # Call OpenAI API
            raw_response = openai_client.beta.chat.completions.parse(
                model=args.model_name_or_path,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url":  {
                                    "url": image_base64
                                }
                            },
                        ],
                    }
                ],
                **generate_kwargs)

            response = raw_response.choices[0].message.content
            print("Generated text:", response)
            action, action_obj, format_reward, action_string = varify_generated_text_func(response)
            return ActionSelectionResult(
                action=action,
                format_reward=format_reward,
                generated_text=response,
                unsupported_action=False,
            )

    elif args.model_type == "openai-computer-use":
        # OpenAI Computer Use API
        # --eval_episodes 20 --model_type openai-computer-use
        import openai
        from baselines.openai_adapter import handle_model_action
        openai_client = openai.Client(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
        user_prompt = "Your goal is to explore this environment as much as possible within a limited number of steps. Please select a meaningful action to continue exploring. Note that opening icons on the desktop requires a double click. You must only use mouse and keyboard inputs. No other tools or input devices are permitted."

        def select_action(one_obs):
            image = Image.fromarray(one_obs, mode="RGB")
            image_base64 = convert_image_to_base64(image)

            # Send a CUA request
            response = openai_client.responses.create(
                model="computer-use-preview",
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": args.video_width,
                    "display_height": args.video_height,
                    "environment": "linux"
                }],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": image_base64
                            }
                        ]
                    }
                ],
                reasoning={
                    "summary": "concise",
                },
                truncation="auto"
            )

            print("OpenAI computer use preview response:", response.output)

            action = None
            action_dict = None
            action_string = None
            format_reward = True
            unsupported_action = False

            for item in response.output:
                if item.type == "computer_call":
                    action_dict = item.action
                    break

            if action_dict is not None:
                try:
                    action_string = handle_model_action(action_dict)
                    action = parse_action(action_string, "discrete", args.video_width, args.video_height)
                    action = action_dict_covert_to_numpy(action)
                except UnSupportedActionError as e:
                    format_reward = False
                    action_string = None
                    unsupported_action = True
                except FormatError as e:
                    format_reward = False
                    action_string = None
                    unsupported_action = True
            else:
                format_reward = False
                action_string = None
                unsupported_action = True

            return ActionSelectionResult(
                action=action,
                format_reward=format_reward,
                generated_text=str(response.output),
                unsupported_action=unsupported_action
            )
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Run evaluation episodes
    for episode in range(args.eval_episodes):
        print(f"Starting evaluation episode {episode+1}/{args.eval_episodes}")
        episode_start_time = time.time()

        # Reset environments
        env.reset()

        # Start recording
        env.start_recording(path=output_dir, record_extra_info=True)

        # Run steps
        for step in tqdm(range(args.max_steps), desc=f"Episode {episode+1}"):
            # Get observations
            before_obs = env.render()

            action_selection_result = select_action(before_obs)
            action = action_selection_result.action
            format_reward = action_selection_result.format_reward
            generated_text = action_selection_result.generated_text
            unsupported_action = action_selection_result.unsupported_action

            if unsupported_action or (not format_reward):
                # Use empty action if format is invalid
                selected_action = create_empty_action_v4(args.mouse_coordinates_type)
            else:
                selected_action = action

            # Execute actions
            after_obs, rewards, terminated, truncated, info = env.step(selected_action)

            before_image_tokens = tokenize_image_func([before_obs])
            after_image_tokens = tokenize_image_func([after_obs])

            env.record_extra_info(step_id=step, data={
                "generated_text": generated_text,
                "after_image_tokens": convert_tensor_to_numpy(after_image_tokens[0]),
                "before_image_tokens": convert_tensor_to_numpy(before_image_tokens[0]),
            })

            if terminated:
                break

        # Stop recording
        env.stop_recording()

    # Close environments
    env.close()
    print(f"Evaluation episode {episode+1} completed in {time.time() - episode_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
