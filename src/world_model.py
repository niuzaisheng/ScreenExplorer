from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from einops import rearrange

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaConfig

class LlamaForWorldModelConfig(LlamaConfig):
    def __init__(self, 
                 use_image = True,
                 image_token_dim=16, 
                 use_text = True,
                 text_embedding_dim=1024, 
                 **kwargs):
        super().__init__(**kwargs)

        self.use_image = use_image
        self.image_token_dim = image_token_dim

        self.use_text = use_text
        self.text_embedding_dim = text_embedding_dim

        assert self.use_image or self.use_text, "At least one of use_image or use_text must be True."


class LlamaForWorldModel(LlamaPreTrainedModel):

    def __init__(self, config:LlamaForWorldModelConfig):
        super().__init__(config)

        self.model = LlamaModel(config)
        self.hidden_size = config.hidden_size
        self.torch_dtype = config.torch_dtype
        self.use_image = config.use_image
        self.use_text = config.use_text

        if self.use_image:
            self.input_proj = nn.Sequential(
                nn.Linear(config.image_token_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )

            self.image_output_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, config.image_token_dim)
            )

        if self.use_text:
            self.text_embedding_input_proj = nn.Sequential(
                nn.Linear(config.text_embedding_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            
            self.text_embedding_output_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, config.text_embedding_dim)
            )

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor,  # [batch_size, seq_len]
        attention_mask: torch.LongTensor,  # [batch_size, seq_len]
        input_image_tokens: torch.FloatTensor = None,  # [batch_size, grid_H, grid_W, image_token_dim]
        input_text_embeddings: torch.FloatTensor = None,  # [batch_size, text_embedding_dim]
        label_image_tokens: torch.FloatTensor = None,  # [batch_size, grid_H, grid_W, image_token_dim]
        label_text_embeddings: torch.FloatTensor = None,  # [batch_size, text_embedding_dim]
        **kwargs,
    ):
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        frame_token_num = 0

        token_inputs_embeds = self.model.get_input_embeddings()(input_ids) # [batch_size, seq_len, hidden_size]
        
        if self.use_image:
            batch_size, grid_H, grid_W, image_token_dim = input_image_tokens.shape
            frame_token_num += grid_H * grid_W # +1 for the grathered text embedding token
            input_image_tokens = rearrange(input_image_tokens, 'B H W D -> B (H W) D')  # [batch_size, frame_token_num, image_token_dim]
            input_image_embeds = self.input_proj(input_image_tokens)  # [batch_size, frame_token_num, hidden_size]
            
        
        if self.use_text:
            frame_token_num += 1 # for the grathered text embedding token
            input_text_embeddings = input_text_embeddings.to(self.torch_dtype)
            input_text_embeddings = self.text_embedding_input_proj(input_text_embeddings)  # [batch_size, hidden_size]
            input_text_embeddings = input_text_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
    
        if self.use_image and self.use_text:
            inputs_embeds = torch.cat([token_inputs_embeds, input_text_embeddings, input_image_embeds], dim=1)  # [batch_size, seq_len+frame_token_num, hidden_size]
        elif self.use_image and not self.use_text:
            inputs_embeds = torch.cat([token_inputs_embeds, input_image_embeds], dim=1)
        elif not self.use_image and self.use_text:
            inputs_embeds = torch.cat([token_inputs_embeds, input_text_embeddings], dim=1)
        else:
            raise ValueError("At least one of use_image or use_text must be True.")
        
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, frame_token_num, device=device)], dim=1)  # [batch_size, block_token_num+seq_len]
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        h = outputs.last_hidden_state  # [batch_size, block_token_num+seq_len, hidden_size]

        if self.use_image and self.use_text:
            # obs reconstruction loss
            h_frame = h[:, -frame_token_num:, :]  # [batch_size, frame_token_num, hidden_size]
            h_text_embedding = h_frame[:, 0, :]  # [batch_size, hidden_size]
            h_obs = h_frame[:, 1:, :]  # [batch_size, frame_token_num-1, hidden_size]
        elif self.use_image and not self.use_text:
            # obs reconstruction loss
            h_obs = h[:, -frame_token_num:, :]
            h_text_embedding = None
        elif not self.use_image and self.use_text:
            # obs reconstruction loss
            h_obs = None
            h_text_embedding = h[:, -frame_token_num:, :]

        pred_obs = None
        if self.use_image:
            pred_obs = self.image_output_proj(h_obs)  # [batch_size, time_step, frame_token_num, image_token_dim]
            pred_obs = rearrange(pred_obs, 'B (H W) D -> B H W D', H=grid_H, W=grid_W)  # [batch_size, time_step, grid_H, grid_W, image_token_dim]
            pred_obs = pred_obs.detach()

        pred_text_embeddings = None
        if self.use_text:
            pred_text_embeddings = self.text_embedding_output_proj(h_text_embedding)  # [batch_size, text_embedding_dim]
                    
        # If labels are provided, calculate losses
        losses = {}
        loss = None
        if label_image_tokens is not None or label_text_embeddings is not None:
            loss = torch.tensor(0.0, device=device)
            
            # Calculate image losses if enabled and labels provided
            if self.use_image and label_image_tokens is not None:
                obs_recon_loss = F.mse_loss(pred_obs, label_image_tokens)
                losses["obs_recon_loss"] = obs_recon_loss
                loss += obs_recon_loss
                
            # Calculate text losses if enabled and labels provided
            if self.use_text and label_text_embeddings is not None:
                label_text_embeddings = label_text_embeddings.to(self.torch_dtype)
                text_embedding_recon_loss = F.mse_loss(pred_text_embeddings, label_text_embeddings)
                losses["text_embedding_recon_loss"] = text_embedding_recon_loss
                loss += text_embedding_recon_loss
                
            # Only set loss attributes if we have any losses
            if losses:
                loss = loss
                losses = {key: losses[key].detach() for key in losses}
                
        return LlamaForWorldModelOutput(
            loss=loss,
            losses=losses,
            pred_obs=pred_obs,
            pred_text_embeddings=pred_text_embeddings,
        )
    

@dataclass
class LlamaForWorldModelOutput(CausalLMOutputWithPast):

    loss: Optional[torch.FloatTensor] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None
    pred_obs: Optional[torch.FloatTensor] = None
    pred_text_embeddings: Optional[torch.FloatTensor] = None