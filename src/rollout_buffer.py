import os
import json
from datetime import datetime
import numpy as np

import torch
from typing import Any, NamedTuple, Union, List, Optional
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import random

class RolloutBuffer:
    """Buffer for storing actor model related data. For vllm"""

    def __init__(self,
                 actor_model_padding_token_id,
                 world_model_padding_token_id=None,
                 use_world_model=True,
                 use_image=True,
                 use_text=True,
                 device="cpu"):
        self.actor_model_padding_token_id = actor_model_padding_token_id
        self.world_model_padding_token_id = world_model_padding_token_id
        self.use_world_model = use_world_model
        self.use_image = use_image
        self.use_text = use_text
        self.device = device
        self.start_time = None
        self.reset()

    def reset(self) -> None:
        self.frame_id = []
        self.episode_id = []
        self.env_id = []
        self.step = []

        # add by actor model
        self.action_model_input_ids = []
        self.action_model_completion_mask = []
        self.format_rewards = []
        self.rewards = []
        self.advantages = []
        self.before_pixel_values = []
        self.after_pixel_values = []
        self.generated_text = []
        self.action_dict = []
        self.action_obj = []
        self.intent_string = []
        self.action_string = []

        # add by world model
        if self.use_world_model:
            self.world_model_input_ids = []
            if self.use_image:
                self.pred_obs = []
            if self.use_text:
                self.pred_text_embeddings = []

        self.before_image_tokens = []
        self.after_image_tokens = []
        self.before_text_embeddings = []
        self.after_text_embeddings = []
        self.intent_string_embeddings = []
        self.before_screen_ocr_results = []
        self.after_screen_ocr_results = []

    @property
    def buffer_size(self):
        return len(self.frame_id)

    def _get_item_stats(self, stat, tensor):
        if isinstance(tensor, list):
            try:
                tensor = torch.stack(tensor, dim=0)
            except:
                return {}  # Skip if can't be stacked
        return {
            f"{stat}_mean": tensor.mean().item(),
            f"{stat}_max": tensor.max().item(),
            f"{stat}_min": tensor.min().item(),
        }

    def _get_stat_keys(self):
        return ["format_rewards", "rewards", "advantages"]

    def get_group_stats(
        self,
            episode_id: int = None,
            env_id: int = None,
            step: int = None
    ):
        assert episode_id is not None or env_id is not None or step is not None, "At least one of episode_id, env_id, or step must be provided."

        if episode_id is not None:
            indices = [i for i, e in enumerate(self.episode_id) if e == episode_id]
        elif env_id is not None:
            indices = [i for i, e in enumerate(self.env_id) if e == env_id]
        elif step is not None:
            indices = [i for i, e in enumerate(self.step) if e == step]
        if len(indices) == 0:
            raise ValueError(f"No samples found for episode_id {episode_id}, env_id {env_id}, or step {step}.")
        stats_dict = {}
        for stat in self._get_stat_keys():
            if stat in self.__dict__ and self.__dict__[stat] is not None:
                stats_dict.update(self._get_item_stats(stat, [self.__dict__[stat][i] for i in indices]))
        return stats_dict

    def add_by_actor(
        self,
        episode_id: List[Union[int, str]],
        env_id: List[Union[int, str]],
        step: List[int],
        batch_size=1,
        **kwargs
    ):
        if self.start_time is None:
            self.start_time = datetime.now()

        for i in range(batch_size):
            if isinstance(episode_id, list):
                item_episode_id = episode_id[i]
            else:
                item_episode_id = episode_id
            if isinstance(env_id, list):
                item_env_id = env_id[i]
            else:
                item_env_id = env_id
            if isinstance(step, list):
                item_step = step[i]
            else:
                item_step = step
            self.frame_id.append(f"{item_episode_id}_{item_env_id}_{item_step}")
            self.episode_id.append(item_episode_id)
            self.env_id.append(item_env_id)
            self.step.append(item_step)

            prompt_ids = kwargs["prompt_ids"][i]
            completion_ids = kwargs["completion_ids"][i]
            action_model_input_ids = prompt_ids + completion_ids
            self.action_model_input_ids.append(action_model_input_ids)
            action_model_completion_mask = [False] * len(prompt_ids) + [True] * len(completion_ids)
            self.action_model_completion_mask.append(action_model_completion_mask)

            self.format_rewards.append(kwargs["format_rewards"][i])
            self.rewards.append(None)
            self.advantages.append(None)
            self.before_pixel_values.append(kwargs["before_pixel_values"][i])
            self.after_pixel_values.append(kwargs["after_pixel_values"][i])
            self.generated_text.append(kwargs["generated_text"][i])
            self.action_dict.append(kwargs["action_dict"][i])
            self.action_obj.append(kwargs["action_obj"][i])
            self.action_string.append(kwargs["action_string"][i])
            self.intent_string.append(kwargs["intent_string"][i])
            self.before_screen_ocr_results.append(kwargs["before_screen_ocr_results"][i])
            self.after_screen_ocr_results.append(kwargs["after_screen_ocr_results"][i])

            if self.use_world_model:
                self.world_model_input_ids.append(None)
                if self.use_image:
                    self.pred_obs.append(None)
                if self.use_text:
                    self.pred_text_embeddings.append(None)

            self.before_image_tokens.append(None)
            self.after_image_tokens.append(None)
            self.before_text_embeddings.append(None)
            self.after_text_embeddings.append(None)
            self.intent_string_embeddings.append(None)


    def get_batch_for_actor(self, batch_size=32):
        """Generate batches for actor model training"""
        # Randomly shuffle indices for actor training (exploration)
        indices = list(range(self.buffer_size))
        random.shuffle(indices)

        class BatchIterator:
            def __init__(self, buffer, indices, batch_size, total_samples):
                self.buffer = buffer
                self.indices = indices
                self.batch_size = batch_size
                self.total_samples = total_samples

            def __len__(self):
                return (self.total_samples + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                start_idx = 0
                while start_idx < self.total_samples:
                    end_idx = min(start_idx + self.batch_size, self.total_samples)
                    yield self.buffer._get_samples_for_actor(self.indices[start_idx:end_idx])
                    start_idx += self.batch_size

        return BatchIterator(self, indices, batch_size, len(indices))

    def _get_samples_for_actor(self, indices):
        # Ensure all advantages have been computed
        for i in indices:
            assert self.advantages[i] is not None, f"Episode {self.group_id[i]} not computed advantages yet."

        # Convert lists of lists to torch tensors first
        input_ids_tensors = []
        attention_mask_tensors = []
        completion_mask_tensors = []

        for i in indices:
            input_ids = torch.tensor(self.action_model_input_ids[i],
                                     dtype=torch.long,
                                     device=self.device)
            input_ids_tensors.append(input_ids)
            attention_mask_tensors.append(torch.ones_like(input_ids, dtype=torch.long, device=self.device))
            completion_mask_tensors.append(torch.tensor(self.action_model_completion_mask[i],
                                                        dtype=torch.bool,
                                                        device=self.device))

        action_model_input_ids = pad_sequence(
            input_ids_tensors,
            batch_first=True,
            padding_value=self.actor_model_padding_token_id,
            padding_side="left"
        )
        action_model_attention_mask = pad_sequence(
            attention_mask_tensors,
            batch_first=True,
            padding_value=0,
            padding_side="left"
        )
        action_model_completion_mask = pad_sequence(
            completion_mask_tensors,
            batch_first=True,
            padding_value=False,
            padding_side="left"
        )

        return ActorModelBufferSamples(
            action_model_input_ids=action_model_input_ids,
            action_model_attention_mask=action_model_attention_mask,
            action_model_completion_mask=action_model_completion_mask,
            advantages=torch.stack([self.advantages[i] for i in indices], dim=0),
            before_pixel_values=[self.before_pixel_values[i] for i in indices],
        )

    # preparing_features START

    def get_batch_for_preparing_features(self, batch_size=32):
        indices = [i for i in range(self.buffer_size) if self.format_rewards[i] == True]

        class BatchIterator:
            def __init__(self, buffer, indices, batch_size, total_samples):
                self.buffer = buffer
                self.indices = indices
                self.batch_size = batch_size
                self.total_samples = total_samples

            def __len__(self):
                return (self.total_samples + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                start_idx = 0
                while start_idx < self.total_samples:
                    end_idx = min(start_idx + self.batch_size, self.total_samples)
                    yield self.buffer._get_samples_for_preparing_features(self.indices[start_idx:end_idx])
                    start_idx += self.batch_size

        return BatchIterator(self, indices, batch_size, len(indices))

    def _get_samples_for_preparing_features(self, indices):
        return indices, PrepareFeatureSamples(
            intent_string=[self.intent_string[i] for i in indices],
            before_pixel_values=[self.before_pixel_values[i] for i in indices],
            after_pixel_values=[self.after_pixel_values[i] for i in indices],
            before_screen_ocr_results=[self.before_screen_ocr_results[i] for i in indices],
            after_screen_ocr_results=[self.after_screen_ocr_results[i] for i in indices], 
        )

    def update_prepared_samples_features(self, indices, **kwargs):
        convert_info = {
            "before_image_tokens": torch.float,
            "after_image_tokens": torch.float,
            "before_text_embeddings": torch.float,
            "after_text_embeddings": torch.float,
            "intent_string_embeddings": torch.float,
        }

        for key in convert_info:
            assert key in kwargs, f"Key {key} not found in kwargs."
            value = kwargs[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=convert_info[key], device=self.device)
            if value.device != self.device:
                value = value.to(self.device)
            kwargs[key] = value

        # 选择那些format_rewards==true的样本，生成world model的输入
        for i, buffer_index in enumerate(indices):
            self.before_image_tokens[buffer_index] = kwargs["before_image_tokens"][i]
            self.after_image_tokens[buffer_index] = kwargs["after_image_tokens"][i]
            self.before_text_embeddings[buffer_index] = kwargs["before_text_embeddings"][i]
            self.after_text_embeddings[buffer_index] = kwargs["after_text_embeddings"][i]
            self.intent_string_embeddings[buffer_index] = kwargs["intent_string_embeddings"][i]
    # preparing_features END

    # preparing_world_model START
    def get_batch_for_preparing_world_model(self, batch_size=32):
        assert self.use_world_model, "World model is not enabled."
        indices = [i for i in range(self.buffer_size) if self.format_rewards[i] == True]

        class BatchIterator:
            def __init__(self, buffer, indices, batch_size, total_samples):
                self.buffer = buffer
                self.indices = indices
                self.batch_size = batch_size
                self.total_samples = total_samples

            def __len__(self):
                return (self.total_samples + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                start_idx = 0
                while start_idx < self.total_samples:
                    end_idx = min(start_idx + self.batch_size, self.total_samples)
                    yield self.buffer._get_samples_for_preparing_world_model(self.indices[start_idx:end_idx])
                    start_idx += self.batch_size

        return BatchIterator(self, indices, batch_size, len(indices))

    def _get_samples_for_preparing_world_model(self, indices):
        return indices, WorldModelSamplesWitingForPrepare(
            action_string=[self.action_string[i] for i in indices],
        )

    def update_prepared_world_model_inputs(self, indices, **kwargs):
        assert self.use_world_model, "World model is not enabled."
        for i, buffer_index in enumerate(indices):
            world_model_input_ids = kwargs["world_model_input_ids"][i]
            world_model_attention_mask = kwargs["world_model_attention_mask"][i].bool()
            world_model_input_ids = world_model_input_ids[world_model_attention_mask]
            self.world_model_input_ids[buffer_index] = world_model_input_ids

    def get_batch_for_world_model(self, batch_size=32):
        """
            Generate batches for world model training
        """
        assert self.use_world_model, "World model is not enabled."
        # Filter indices where format_rewards is True
        indices = [i for i in range(self.buffer_size) if self.format_rewards[i] == True]
        random.shuffle(indices)

        class BatchIterator:
            def __init__(self, buffer, indices, batch_size, total_samples):
                self.buffer = buffer
                self.indices = indices
                self.batch_size = batch_size
                self.total_samples = total_samples

            def __len__(self):
                return (self.total_samples + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                start_idx = 0
                while start_idx < self.total_samples:
                    end_idx = min(start_idx + self.batch_size, self.total_samples)
                    yield self.buffer._get_samples_for_world_model(self.indices[start_idx:end_idx])
                    start_idx += self.batch_size

        return BatchIterator(self, indices, batch_size, len(indices))

    def _get_samples_for_world_model(self, indices):
        world_model_input_ids = pad_sequence(
            [self.world_model_input_ids[i] for i in indices],
            batch_first=True,
            padding_value=self.world_model_padding_token_id,
            padding_side="left"
        )
        world_model_attention_mask = world_model_input_ids != self.world_model_padding_token_id

        return indices, WorldModelBufferSamples(
            world_model_input_ids=world_model_input_ids,
            world_model_attention_mask=world_model_attention_mask,
            before_image_tokens=torch.stack([self.before_image_tokens[i] for i in indices], dim=0),
            after_image_tokens=torch.stack([self.after_image_tokens[i] for i in indices], dim=0),
            before_text_embeddings=torch.stack([self.before_text_embeddings[i] for i in indices], dim=0),
            after_text_embeddings=torch.stack([self.after_text_embeddings[i] for i in indices], dim=0),
        )

    def update_world_model_output(self, indices, **kwargs):
        assert self.use_world_model, "World model is not enabled."
        convert_info = {}
        if self.use_image:
            convert_info["pred_obs"] = torch.float
        if self.use_text:
            convert_info["pred_text_embeddings"] = torch.float

        for key in convert_info:
            assert key in kwargs, f"Key {key} not found in kwargs."
            value = kwargs[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=convert_info[key], device=self.device)
            if value.device != self.device:
                value = value.to(self.device)
            kwargs[key] = value

        for i, buffer_index in enumerate(indices):
            if self.use_image:
                self.pred_obs[buffer_index] = kwargs["pred_obs"][i]
            if self.use_text:
                self.pred_text_embeddings[buffer_index] = kwargs["pred_text_embeddings"][i]
    # preparing_world_model END

    # reward_compute START
    def get_samples_for_reward_compute(self):
        batch_indexes = defaultdict(list)
        for i, (episode_id, env_id) in enumerate(zip(self.episode_id, self.env_id)):
            if self.format_rewards[i] == True:
                batch_indexes[f"{env_id}_{episode_id}"].append(i)

        class BatchIterator:
            def __init__(self, buffer, batch_indexes, total_samples):
                self.buffer = buffer
                self.batch_indexes = batch_indexes
                self.total_samples = total_samples

            def __len__(self):
                return self.total_samples

            def __iter__(self):
                for batch in self.batch_indexes.values():
                    yield self.buffer._get_samples_for_reward_compute(batch)

        return BatchIterator(self, batch_indexes, len(batch_indexes.keys()))

    def _get_samples_for_reward_compute(self, indices):
        return indices, RewardComputeSamples(
            format_rewards=torch.tensor([self.format_rewards[i] for i in indices], dtype=torch.bool, device=self.device),
            before_image_tokens=torch.stack([self.before_image_tokens[i] for i in indices], dim=0),
            after_image_tokens=torch.stack([self.after_image_tokens[i] for i in indices], dim=0),
            pred_after_image_tokens=torch.stack([self.pred_obs[i] for i in indices], dim=0) if self.use_world_model and self.use_image else None,
            before_text_embeddings=torch.stack([self.before_text_embeddings[i] for i in indices], dim=0),
            after_text_embeddings=torch.stack([self.after_text_embeddings[i] for i in indices], dim=0),
            pred_text_embeddings=torch.stack([self.pred_text_embeddings[i] for i in indices], dim=0) if self.use_world_model and self.use_text else None,
            action_dict=[self.action_dict[i] for i in indices],
            before_screen_ocr_results=[self.before_screen_ocr_results[i] for i in indices],
            intent_string_embeddings=torch.stack([self.intent_string_embeddings[i] for i in indices], dim=0),
        )

    def update_reward(self, indices, rewards):
        for i, buffer_index in enumerate(indices):
            self.rewards[buffer_index] = rewards[i]

    def get_all_rewards(self):
        for i in range(self.buffer_size):
            if self.format_rewards[i] == False:
                self.rewards[i] = torch.tensor(0.0, dtype=torch.float, device=self.device)

        self.format_rewards = torch.tensor(self.format_rewards, dtype=torch.float, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float, device=self.device)
        self.rewards = rewards * self.format_rewards
        return self.rewards

    def update_advantages(self, advantages):
        self.advantages = advantages
    # reward_compute END

    def save_recording(self, recording_dir):

        format_rewards = self.format_rewards.cpu().tolist()
        rewards = self.rewards.cpu().tolist()
        advantages = self.advantages.cpu().tolist()
        all_records = []
        for index in range(self.buffer_size):
            all_records.append({
                "frame_id": self.frame_id[index],
                "episode_id": self.episode_id[index],
                "env_id": self.env_id[index],
                "step": self.step[index],
                "format_rewards": format_rewards[index],
                "rewards": rewards[index],
                "advantages": advantages[index],
                "generated_text": self.generated_text[index],
                "action_string": self.action_string[index],
                "intent_string": self.intent_string[index],
                "before_screen_ocr_results": self.before_screen_ocr_results[index],
                "after_screen_ocr_results": self.after_screen_ocr_results[index],
            })

        all_records_dict = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "buffer_size": self.buffer_size,
            "all_records": all_records
        }

        os.makedirs(os.path.dirname(recording_dir), exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(recording_dir, f"game_record_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_records_dict, f, ensure_ascii=False, indent=2)

        return filename


class PrepareFeatureSamples(NamedTuple):
    intent_string: List[str]
    before_pixel_values: List[np.ndarray]
    after_pixel_values: List[np.ndarray]
    before_screen_ocr_results: List[Any]
    after_screen_ocr_results: List[Any]


class WorldModelSamplesWitingForPrepare(NamedTuple):
    action_string: List[str]


class WorldModelBufferSamples(NamedTuple):
    world_model_input_ids: torch.LongTensor
    world_model_attention_mask: torch.BoolTensor
    before_image_tokens: torch.FloatTensor
    after_image_tokens: torch.FloatTensor
    before_text_embeddings: Optional[torch.FloatTensor]
    after_text_embeddings: Optional[torch.FloatTensor]

    def to(self, target_device):
        return self._replace(**{key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                                for key, value in self._asdict().items()})


class RewardComputeSamples(NamedTuple):
    format_rewards: torch.BoolTensor
    before_image_tokens: torch.FloatTensor
    after_image_tokens: torch.FloatTensor
    pred_after_image_tokens: torch.FloatTensor
    before_text_embeddings: Optional[List[Any]]
    after_text_embeddings: Optional[List[Any]]
    pred_text_embeddings: Optional[List[Any]]
    action_dict: Optional[List[Any]]
    before_screen_ocr_results: Optional[List[Any]]
    intent_string_embeddings: Optional[List[Any]]

    def to(self, target_device):
        return self._replace(**{key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                                for key, value in self._asdict().items()})


class ActorModelBufferSamples(NamedTuple):
    action_model_input_ids: torch.LongTensor
    action_model_attention_mask: torch.BoolTensor
    action_model_completion_mask: torch.BoolTensor
    advantages: torch.FloatTensor
    before_pixel_values: List[np.ndarray]

    def to(self, target_device):
        return self._replace(**{key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                                for key, value in self._asdict().items()})
