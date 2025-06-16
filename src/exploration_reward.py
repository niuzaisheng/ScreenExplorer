import torch
import torch.nn.functional as F
from typing import Any, NamedTuple, Union, List, Optional, Dict


def compute_cosine_similarity(images_a, images_b=None):
    if images_b is None:
        n_images = len(images_a)
        flat_a = images_a.reshape(n_images, -1)  # [n_images, H*W*D]
        a_expanded = flat_a.unsqueeze(1)  # [n_images, 1, H*W*D]
        b_expanded = flat_a.unsqueeze(0)  # [1, n_images, H*W*D]
        similarity_matrix = F.cosine_similarity(a_expanded, b_expanded, dim=2)
        return similarity_matrix

    else:
        flat_a = images_a.reshape(images_a.shape[0], -1)
        flat_b = images_b.reshape(images_b.shape[0], -1)
        similarities = F.cosine_similarity(flat_a, flat_b, dim=1)
        return similarities


def compute_sequence_diversity_reward(sim_matrix):
    """
    Calculate sequence diversity reward based on similarity matrix, applicable for complete sequences including initial image

    Args:
        sim_matrix: Similarity matrix [sequence_length, sequence_length],
                    where the first row/column corresponds to the initial image

    Returns:
        torch.Tensor: Sequence diversity reward [sequence_length], including reward for initial image
    """
    n_images = sim_matrix.shape[0]
    rewards = torch.zeros(n_images, device=sim_matrix.device)

    # For other steps, calculate the average difference between past states and future states
    for i in range(1, n_images):
        # Average inverse mutual information between all past states(0:i) and current/future states(i:n_images)
        past_future_block = sim_matrix[:i, i:]
        rewards[i] = torch.mean(past_future_block)

    return rewards


class RewardFlags(NamedTuple):
    immediate: bool = True  # Immediately Change Reward
    sequence: bool = True  # Subsequent Change Reward
    world_model: bool = True  # World Model Reconstructing Reward
    text: bool = True  # Enable text embedding reward
    image: bool = True  # Enable image embedding reward
    intent_ocr: bool = True  # Intent-Action Alignment Reward
    intent_coordinates: bool = True  # Intent-Observation Alignment Reward

    def __str__(self):
        return f"RewardFlags(immediate={self.immediate}, sequence={self.sequence}, world_model={self.world_model}, text={self.text}, image={self.image}, intent_ocr={self.intent_ocr}, intent_coordinates={self.intent_coordinates})"

def parse_reward_function_string(reward_function_str):
    
    # Start with default values
    reward_flags_dict = {
        "immediate": True,
        "sequence": True,
        "world_model": True,
        "text": True,
        "image": True,
        "intent_ocr": True,
        "intent_coordinates": True,
    }
    
    # Handle special keywords
    if reward_function_str.lower() == "all":
        return RewardFlags(**reward_flags_dict)
    
    # Process the component settings
    components = [c.strip() for c in reward_function_str.split(",")]
    
    for component in components:
        if not component:
            continue
        
        if component in reward_flags_dict:
            reward_flags_dict[component] = True

        elif component.startswith("+"):
            name = component[1:].lower()
            if name in reward_flags_dict:
                reward_flags_dict[name] = True
            else:
                raise ValueError(f"Unknown reward component: {name}")
            
        elif component.startswith("no_"):
            name = component[3:].lower()
            if name in reward_flags_dict:
                reward_flags_dict[name] = False
            else:
                raise ValueError(f"Unknown reward component: {name}")
        else:
            raise ValueError(f"Component must start with + or -: {component}")
    
    return RewardFlags(**reward_flags_dict)


@torch.no_grad()
def compute_exploration_rewards_by_abs_value_with_ocr_v7(
    reward_flags:RewardFlags,
    before_image_tokens=None,
    after_image_tokens=None,
    pred_after_image_tokens=None,
    before_text_embeddings=None,
    after_text_embeddings=None,
    pred_text_embeddings=None,
    batch_action_dict=None,
    before_screen_ocr_results=None, 
    intent_string_embeddings=None,
    method="cosine", 
    text_embedding_func=None,
):
    """
    Compute exploration rewards using similarity values
    
    Rewards include:
        - Immediate state change rewards:
            - Image similarity immediate reward
            - Text similarity immediate reward
        - Interaction position rewards:
            - Intent to bbox text similarity reward
        - Sequence rewards:
            - Image sequence reward
            - Text sequence reward
        - World model rewards:
            - Image world model reward
            - Text world model reward

    Args:
        text_embedding_func: Text embedding function
        before_image_tokens: Images before action execution [batch_size, H, W, D]
        after_image_tokens: Real images after action execution [batch_size, H, W, D]
        pred_after_image_tokens: Predicted images after action [batch_size, H, W, D]
        before_text_embeddings: Text embeddings before action [batch_size, embedding_dim]
        after_text_embeddings: Text embeddings after action [batch_size, embedding_dim]
        pred_text_embeddings: Predicted text embeddings after action [batch_size, embedding_dim]
        batch_action_dict: Action dictionary containing coordinate information
        before_screen_ocr_results: OCR results before action with text and coordinates
        intent_string_embeddings: Text embeddings generated by Actor model [batch_size, embedding_dim]
        method: Similarity method: "cosine", "kl", "wasserstein"

    Returns:
        Dictionary containing similarity values as rewards
    """

    assert text_embedding_func is not None, "text_embedding_func must be provided"
    from screen_env import ActionType

    # Choose similarity calculation method
    if method == "cosine":
        similarity_func = compute_cosine_similarity
    else:
        raise ValueError(f"Unknown method: {method}")

    batch_size = before_image_tokens.shape[0]
    device = before_image_tokens.device

    combined_reward = torch.zeros(batch_size, device=device)
    reward_dict = {}
    # Image comparison
    if reward_flags.image:

        if reward_flags.immediate:
            ## Image similarity immediate reward
            curiosity_sim = similarity_func(before_image_tokens, after_image_tokens)
            combined_reward += (1.0 - curiosity_sim)  # Higher similarity, higher value
            reward_dict["curiosity_sim"] = curiosity_sim

        if reward_flags.sequence:
            ## Image sequence diversity calculation
            first_image = before_image_tokens[0:1]
            complete_sequence = torch.cat([first_image, after_image_tokens], dim=0)
            similarity_matrix = similarity_func(complete_sequence)  # Calculate similarity matrix for complete sequence
            sequence_sim = compute_sequence_diversity_reward(similarity_matrix)  # Calculate sequence state diversity reward
            sequence_sim = sequence_sim[1:]  # Remove initial image reward
            combined_reward += (1.0 - sequence_sim)  # Higher similarity, higher value
            reward_dict["sequence_sim"] = sequence_sim
            reward_dict["similarity_matrix"] = similarity_matrix[1:, 1:]

        if reward_flags.world_model:
            ## Calculate similarity between real and predicted images (World Model derived reward)
            novelty_sim = similarity_func(after_image_tokens, pred_after_image_tokens)
            combined_reward += (1.0 - novelty_sim)  # Higher similarity, higher value
            reward_dict["novelty_sim"] = novelty_sim

    # Text comparison
    if reward_flags.text:
        if reward_flags.immediate:
            ## Text similarity immediate reward
            curiosity_sim_text = similarity_func(before_text_embeddings, after_text_embeddings)
            combined_reward += (1.0 - curiosity_sim_text)
            reward_dict["curiosity_sim_text"] = curiosity_sim_text

        if reward_flags.sequence:
            ## Text sequence diversity calculation
            first_text = before_text_embeddings[0:1]
            complete_sequence_text = torch.cat([first_text, after_text_embeddings], dim=0)
            similarity_matrix_text = similarity_func(complete_sequence_text)
            sequence_sim_text = compute_sequence_diversity_reward(similarity_matrix_text)  # Calculate sequence state diversity reward
            sequence_sim_text = sequence_sim_text[1:]  # Remove initial image reward
            combined_reward += (1.0 - sequence_sim_text)  # Higher similarity, higher value
            reward_dict["sequence_sim_text"] = sequence_sim_text
            reward_dict["similarity_matrix_text"] = similarity_matrix_text[1:, 1:]

        if reward_flags.world_model:
            ## Calculate similarity between real and predicted text (World Model derived reward)
            novelty_sim_text = similarity_func(after_text_embeddings, pred_text_embeddings)
            combined_reward += (1.0 - novelty_sim_text)
            reward_dict["novelty_sim_text"] = novelty_sim_text
    
    if reward_flags.intent_ocr:
        # Calculate similarity between intent text and OCR text
        before_to_agent_generated_text_sim = similarity_func(before_text_embeddings, intent_string_embeddings)
        after_to_agent_generated_text_sim = similarity_func(after_text_embeddings, intent_string_embeddings)
        # Higher similarity means higher reward
        combined_reward += before_to_agent_generated_text_sim + after_to_agent_generated_text_sim
        reward_dict["before_to_agent_generated_text_sim"] = before_to_agent_generated_text_sim
        reward_dict["after_to_agent_generated_text_sim"] = after_to_agent_generated_text_sim
        
    if reward_flags.intent_coordinates:
        ## Calculate similarity between text in OCR bbox at action coordinates and intent text
        text_in_bbox_to_intent_sim = torch.zeros_like(after_to_agent_generated_text_sim)

        hits_index = []
        hits_text = []
        for index, (item_action_dict, item_before_screen_ocr_result) in enumerate(zip(batch_action_dict, before_screen_ocr_results)):
            # Get action coordinates
            if item_action_dict["action_type"] in [ActionType.click.value, ActionType.right_click.value, ActionType.double_click.value]:
                mouse_coordinates_x = item_action_dict["mouse_coordinates_x"]
                mouse_coordinates_y = item_action_dict["mouse_coordinates_y"]

                hit_text = []
                # Find bbox containing action coordinates in OCR results
                for ocr_result in item_before_screen_ocr_result:
                    # Get OCR result coordinates
                    rec_boxes = ocr_result["boxes"]
                    x_min, y_min, x_max, y_max = rec_boxes

                    # Check if action coordinates are within OCR result bbox
                    if (x_min <= mouse_coordinates_x <= x_max) and (y_min <= mouse_coordinates_y <= y_max):
                        hit_text.append(ocr_result["text"].strip())

                if len(hit_text)>0:
                    hit_text = "Screen OCR: " + " ".join(hit_text)
                    hits_text.append(hit_text)
                    hits_index.append(index)

        # Calculate similarity for hit text
        if len(hits_index) > 0:
            hits_text_embeddings = text_embedding_func(hits_text).to(after_image_tokens.device)
            text_in_bbox_to_intent_sim[hits_index] = similarity_func(hits_text_embeddings, intent_string_embeddings[hits_index])
    
        # Higher similarity means higher reward
        combined_reward += text_in_bbox_to_intent_sim
        reward_dict["text_in_bbox_to_intent_sim"] = text_in_bbox_to_intent_sim


    reward_dict["rewards"] = combined_reward
    return reward_dict
