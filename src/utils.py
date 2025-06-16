import base64
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, NamedTuple

import numpy as np
import torch
from torch import Tensor

from PIL import Image
from accelerate.utils import is_peft_model


class ActionSelectionResult(NamedTuple):
    action: Dict[str, Any]
    format_reward: bool
    generated_text: str
    unsupported_action: bool

def convert_image_to_base64(image:Image) -> str:
    assert isinstance(image, Image.Image), "Input must be a PIL Image"
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"

_DTYPE, _DEVICE = torch.bfloat16, "cuda"
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

def numpy2tensor(
    input_image: np.ndarray,
    dtype: torch.dtype = _DTYPE,
    device: str = _DEVICE,
    range_min: int = -1,
) -> torch.Tensor:
    """Converts image(dtype=np.uint8) to `dtype` in range [0..255].

    Args:
        input_image: A batch of images in range [0..255], BxHxWx3 layout.
    Returns:
        A torch.Tensor of layout Bx3xHxW in range [-1..1], dtype.
    """
    ndim = input_image.ndim
    indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
    image = input_image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(dtype).to(device)

def tokenize_image(image_tokenizer, images):
    images_numpy = images
    if isinstance(images, list) or isinstance(images, tuple):
        images_numpy = np.stack(images_numpy)
    if images_numpy.ndim == 3:
        images_numpy = np.expand_dims(images_numpy, axis=0)
    images_tensor = numpy2tensor(images_numpy, dtype=torch.bfloat16, device="cuda")
    (latent,) = image_tokenizer.encode(images_tensor)  # [B, C, H, W] -> [B, 16, H/8, W/8]
    latent = latent.permute(0, 2, 3, 1)  # [B, H/8, W/8, 16]
    return latent


class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulators for a new episode"""
        self.loss_sums = {}
        self.counts = {}

    def add_losses(self, losses_dict):
        """Add a dictionary of losses to the tracker"""
        for key, value in losses_dict.items():
            # Convert tensor to float if needed
            if torch.is_tensor(value):
                value = value.item()

            # Initialize entry if not exists
            if key not in self.loss_sums:
                self.loss_sums[key] = 0.0
                self.counts[key] = 0

            self.loss_sums[key] += value
            self.counts[key] += 1

    def add_list_dict(self, list_dict: Dict[str, Tensor]):
        
        """Add a dictionary of losses to the tracker"""
        for key, vector in list_dict.items():
            # Initialize entry if not exists
            assert isinstance(vector, Tensor), f"Expected tensor, got {type(vector)}"
            assert vector.ndim == 1, f"Expected 1D tensor, got {vector.ndim}D tensor"
            vector = vector.cpu().tolist()
            if key not in self.loss_sums:
                self.loss_sums[key] = 0.0
                self.counts[key] = 0
            # Add the sum of the vector to the loss sums
            self.loss_sums[key] += sum(vector)
            self.counts[key] += len(vector)

    def get_average_losses(self):
        """Return dictionary of average losses"""
        return {k: self.loss_sums[k] / self.counts[k] for k in self.loss_sums}
    

class GameRecorder:

    def __init__(self, path):
        self.path = path
        self.start_time = datetime.now()
        self.steps = {}
        
    def record_step(self, step_number, **kwargs):
        """Record information for each step"""
        step_data = {
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        if step_number in self.steps:
            self.steps[step_number].update(step_data)
        else:
            self.steps[step_number] = step_data
    
    def save(self):
        """Save record to JSON file"""
        record = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_steps": len(self.steps),
            "steps": self.steps
        }

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.path, f"game_record_{timestamp}.json")        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        return filename
    

def sync_pytorch_to_vllm(model, vllm_model):
    """
    Synchronize parameters from PyTorch model to VLLM model
    """
    vllm_model_core = vllm_model.llm_engine.model_executor.driver_worker.worker.model_runner.model
    if is_peft_model(model):
        # Handle PEFT model case
        model.merge_adapter()
        for name, param in model.named_parameters():
            # When using PEFT, we need to recover the original parameter name and discard some parameters
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if model.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = name.replace("modules_to_save.default.", "")
            vllm_model_core.load_weights(weights=[(name, param)])

    else:
        state_dict = model.state_dict()
        vllm_model_core.load_weights(state_dict.items())

    # Ensure VLLM model updates its internal representations
    vllm_model.reset_prefix_cache()
    print("Synchronized PyTorch model parameters to VLLM")
