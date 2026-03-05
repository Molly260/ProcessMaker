import argparse
import copy
import math
import random
from typing import Any
import pdb
import os
import ast

import time
from PIL import Image, ImageOps

import torch
from accelerate import Accelerator
from library.device_utils import clean_memory_on_device
from safetensors.torch import load_file
from networks import lora_flux

from library import flux_models, flux_train_utils_recraft as flux_train_utils, flux_utils, sd3_train_utils, \
    strategy_base, strategy_flux, train_util
from torchvision import transforms
import train_network
from library.utils import setup_logging
from diffusers.utils import load_image
import numpy as np

import sys
try:
    from sliding_window_utils import sliding_window_detection, compute_window_difference
    SLIDING_WINDOW_AVAILABLE = True
except ImportError:
    SLIDING_WINDOW_AVAILABLE = False

setup_logging()
import logging

logger = logging.getLogger(__name__)



def prepare_text_encoder_fp8(index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0:  # CLIP-L
        logger.info(f"prepare CLIP-L for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
        text_encoder.to(te_weight_dtype)  # fp8
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)
    else:  # T5XXL

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)

                    hidden_states = module.wo(hidden_states)
                    return hidden_states

                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        if flux_utils.get_t5xxl_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
            logger.info(f"T5XXL already prepared for fp8")
        else:
            logger.info(f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
            text_encoder.to(te_weight_dtype)  # fp8
            prepare_fp8(text_encoder, weight_dtype)


def load_target_model(
        fp8_base: bool,
        pretrained_model_name_or_path: str,
        disable_mmap_load_safetensors: bool,
        clip_l_path: str,
        fp8_base_unet: bool,
        t5xxl_path: str,
        ae_path: str,
        weight_dtype: torch.dtype,
        accelerator: Accelerator
):
    # Determine the loading data type
    loading_dtype = None if fp8_base else weight_dtype

    # Load the main model to the accelerator's device
    _, model = flux_utils.load_flow_model(
        pretrained_model_name_or_path,
        # loading_dtype,
        torch.float8_e4m3fn,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )

    if fp8_base:
        # Check dtype of the model
        if model.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
        elif model.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 FLUX model")

    # Load the CLIP model to the accelerator's device
    clip_l = flux_utils.load_clip_l(
        clip_l_path,
        weight_dtype,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )
    clip_l.eval()

    # Determine the loading data type for T5XXL
    if fp8_base and not fp8_base_unet:
        loading_dtype_t5xxl = None  # as is
    else:
        loading_dtype_t5xxl = weight_dtype

    # Load the T5XXL model to the accelerator's device
    t5xxl = flux_utils.load_t5xxl(
        t5xxl_path,
        loading_dtype_t5xxl,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )
    t5xxl.eval()

    if fp8_base and not fp8_base_unet:
        # Check dtype of the T5XXL model
        if t5xxl.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
        elif t5xxl.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 T5XXL model")

    # Load the AE model to the accelerator's device
    ae = flux_utils.load_ae(
        ae_path,
        weight_dtype,
        # accelerator.device,  # Changed from "cpu" to accelerator.device
        "cpu",
        disable_mmap=disable_mmap_load_safetensors
    )

    # # Wrap models with Accelerator for potential distributed setups
    # model, clip_l, t5xxl, ae = accelerator.prepare(model, clip_l, t5xxl, ae)

    return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model


import torchvision.transforms as transforms


class ResizeWithPadding:
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image or a NumPy array")

        width, height = img.size

        if width == height:
            img = img.resize((self.size, self.size), Image.LANCZOS)
        else:
            max_dim = max(width, height)

            new_img = Image.new("RGB", (max_dim, max_dim), (self.fill, self.fill, self.fill))
            new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))

            img = new_img.resize((self.size, self.size), Image.LANCZOS)

        return img


def parse_grid_image(image_path, dataset_frame_num):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    if dataset_frame_num == 4:
        # 2x2 
        rows, cols = 2, 2
        cell_width = width // 2
        cell_height = height // 2
    elif dataset_frame_num == 9:
        # 3x3 
        rows, cols = 3, 3
        cell_width = width // 3
        cell_height = height // 3
    else:
        raise ValueError(f"Unsupported dataset_frame_num: {dataset_frame_num}")
    
    frames = []
    for row in range(rows):
        if row % 2 == 0:
            col_range = range(cols)
        else:
            col_range = range(cols - 1, -1, -1)
        
        for col in col_range:
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            cell_image = image.crop((left, top, right, bottom))
            frames.append(cell_image)
    
    return frames, (rows, cols)


def generate_frame_descriptions(frames, base_prompt):
    
    base_words = base_prompt.split()
    
    
    key_concepts = []
    for word in base_words[:10]:  
        if len(word) > 3 and word.lower() not in ['the', 'and', 'this', 'that', 'with', 'from', 'they', 'have', 'been', 'image', 'drawing']:
            key_concepts.append(word)
        if len(key_concepts) >= 3: 
            break
    
    frame_texts = []
    for i, frame in enumerate(frames):
        if len(key_concepts) > 0:
            frame_desc = f"{' '.join(key_concepts)} step {i+1}"
        else:
            frame_desc = f"construction step {i+1}"
        
        frame_texts.append(frame_desc)

    return frame_texts


def compute_adaptive_k(all_candidates, all_differences, threshold):
   
    if not all_candidates or not all_differences:
        return 0
    
   
    above_threshold = [c for c in all_candidates if c[2] >= threshold]
    
    adaptive_k = len(above_threshold)
   
    adaptive_k = min(adaptive_k, len(all_candidates), 6)
    
    logger.info(f"   - threshold: {threshold:.4f}")
    logger.info(f"   - adaptive_k: {adaptive_k}")
    
    return adaptive_k


def select_frames_with_sliding_window(frames, prompt, alpha=0.6, beta=0.2, device="cuda", adaptive_k=True, adaptive_threshold=0.5):
   
    if not SLIDING_WINDOW_AVAILABLE:
        raise ImportError("Not SLIDING_WINDOW")
    
    prompt_words = prompt.split()
  
    special_chars = sum(1 for c in prompt if c in '.,!?:;<>()[]{}')
    
    if len(prompt_words) > 60 or special_chars > 10:
        texts = generate_frame_descriptions(frames, prompt)
    else:
        texts = [prompt] * len(frames)
    
    pairs, all_differences, threshold = sliding_window_detection(
        frames=frames,
        texts=texts,
        alpha=alpha,
        beta=beta,
        k=0.8,
        device=device
    )
    
    all_candidates = []
    
    if pairs:
        for i, j, score in pairs:
            all_candidates.append((i, j, score, "threshold"))
    
    for i in range(len(frames) - 1):
        if i < len(all_differences):
            if not any(c[0] == i and c[1] == i + 1 for c in all_candidates):
                all_candidates.append((i, i + 1, all_differences[i], "adjacent"))
    
    if len(frames) >= 3:
        for i in range(len(frames) - 2):
            if len(frames) - 1 + i < len(all_differences):
                if not any(c[0] == i and c[1] == i + 2 for c in all_candidates):
                    all_candidates.append((i, i + 2, all_differences[len(frames) - 1 + i], "skip_one"))
    
    if len(frames) >= 2:
        first_last = (0, len(frames) - 1)
        if not any(c[0] == first_last[0] and c[1] == first_last[1] for c in all_candidates):
            score = all_differences[0] if all_differences else 0.0
            all_candidates.append((first_last[0], first_last[1], score, "first_last"))
    
    all_candidates.sort(key=lambda x: x[2], reverse=True)
    
    adaptive_top_k = compute_adaptive_k(all_candidates, all_differences, adaptive_threshold)
    top_candidates = all_candidates[:adaptive_top_k]
    
    selected_pairs = []
    for i, (frame_i, frame_j, score, pair_type) in enumerate(top_candidates):
        if frame_i > frame_j:
            frame_i, frame_j = frame_j, frame_i 
        
        selected_indices = [frame_i, frame_j]
        selected_frames = [frames[frame_i], frames[frame_j]]
        selected_pairs.append([selected_indices, selected_frames, score])
    return selected_pairs


def encode_images_to_latents(args, vae, images):
    # Get image dimensions
    b, c, h, w = images.shape
    num_split = 2 if args.dataset_frame_num == 4 else 3
    # Split the image into multiple parts
    img_parts = [images[:, :, :, i * w // num_split:(i + 1) * w // num_split] for i in range(num_split)]
    # Encode each part
    latents = [vae.encode(img) for img in img_parts]
    # Concatenate latents in the latent space to reconstruct the full image
    latents = torch.cat(latents, dim=-1)
    return latents


def encode_images_to_latents2(args, vae, images):
    latents = vae.encode(images)
    return latents


def sample(args, accelerator, vae, text_encoder, flux, output_dir, sample_images, sample_prompts):
    # Directly use precomputed conditions
    conditions = {}
    with torch.no_grad():
        for image_item, prompt_dict in zip(sample_images, sample_prompts):
            if isinstance(image_item, str):
                try:
                    image_item = ast.literal_eval(image_item)
                  
                except (ValueError, SyntaxError):
                    
                    pass
            
            prompt = prompt_dict.get("prompt", "")
            if prompt not in conditions:
               
                if isinstance(image_item, (list, tuple)) and len(image_item) == 2:
                    init_path, final_path = image_item
                    logger.info(f"Cache conditions for dual images: init={init_path}, final={final_path} with prompt: {prompt}")
                    
                    resize_transform = ResizeWithPadding(size=512, fill=255) if args.dataset_frame_num == 4 else ResizeWithPadding(size=352, fill=255)
                    img_transforms = transforms.Compose([
                        resize_transform,
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])
                    
                    init_image = img_transforms(np.array(load_image(init_path), dtype=np.uint8)).unsqueeze(0).to(
                        vae.device, dtype=vae.dtype
                    )
                    init_latents = encode_images_to_latents2(args, vae, init_image)  
                    
                    final_image = img_transforms(np.array(load_image(final_path), dtype=np.uint8)).unsqueeze(0).to(
                        vae.device, dtype=vae.dtype
                    )
                    final_latents = encode_images_to_latents2(args, vae, final_image)  
                    

                    conditions[prompt] = [init_latents.to("cpu"), final_latents.to("cpu")]
                    
                    logger.debug(f"Encoded dual latents - init: {init_latents.shape}, final: {final_latents.shape}")
                    
                else:
                   
                    image_path = image_item
                    logger.info(f"Processing grid image: {image_path} with prompt: {prompt}")
                    
                   
                    frames, grid_layout = parse_grid_image(image_path, args.dataset_frame_num)
                    
                    if len(frames) < 2:
                        raise ValueError(f"Fail: {len(frames)} < 2")
                    
                    selected_pairs = select_frames_with_sliding_window(
                        frames, prompt, 
                        alpha=getattr(args, 'sliding_window_alpha', 0.6), 
                        beta=getattr(args, 'sliding_window_beta', 0.2), 
                        device=vae.device,
                        adaptive_k=getattr(args, 'adaptive_k', True),
                        adaptive_threshold=getattr(args, 'adaptive_threshold', 0.5)
                    )
                    
                
                    if not selected_pairs:
                        continue
                    
                    resize_transform = ResizeWithPadding(size=512, fill=255) if args.dataset_frame_num == 4 else ResizeWithPadding(size=352, fill=255)
                    img_transforms = transforms.Compose([
                        resize_transform,
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])
                    
                    for pair_idx, (selected_indices, selected_frames, score) in enumerate(selected_pairs):
                        pair_prompt = f"{prompt}_pair{pair_idx}" 
                        
                        init_image = img_transforms(np.array(selected_frames[0], dtype=np.uint8)).unsqueeze(0).to(
                            vae.device, dtype=vae.dtype
                        )
                        init_latents = encode_images_to_latents2(args, vae, init_image) 
                        
                        final_image = img_transforms(np.array(selected_frames[1], dtype=np.uint8)).unsqueeze(0).to(
                            vae.device, dtype=vae.dtype
                        )
                        final_latents = encode_images_to_latents2(args, vae, final_image) 
                        
                        conditions[pair_prompt] = [init_latents.to("cpu"), final_latents.to("cpu")]
                        
                    
                    logger.debug(f"Encoded dual latents from grid - init: {init_latents.shape}, final: {final_latents.shape}")

    sample_conditions = conditions

    if sample_conditions is not None:
        conditions = {k: v for k, v in sample_conditions.items()}  # Already on CUDA

    sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
    text_encoder[0].to(accelerator.device, dtype=torch.bfloat16)  # CLIP-L always use bfloat16
    text_encoder[1].to(accelerator.device)
    
    if text_encoder[1].dtype == torch.float8_e4m3fn:
        # if we load fp8 weights, the model is already fp8, so we use it as is
        prepare_text_encoder_fp8(1, text_encoder[1], text_encoder[1].dtype, torch.bfloat16)
    else:
        # otherwise, we need to convert it to target dtype
        text_encoder[1].to(torch.bfloat16)

    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(512)
    text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(True)

    with accelerator.autocast(), torch.no_grad():
        for prompt_dict in sample_prompts:
            base_prompt = prompt_dict.get("prompt", "")
            negative_prompt = prompt_dict.get("negative_prompt", "")
            
    
            for p in [base_prompt, negative_prompt]:
                if p and p not in sample_prompts_te_outputs:
                    logger.info(f"Cache Text Encoder outputs for prompt: {p}")
                    tokens_and_masks = tokenize_strategy.tokenize(p)
                    sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, text_encoder, tokens_and_masks, True
                    )
     
            pair_prompts = [key for key in conditions.keys() if key.startswith(f"{base_prompt}_pair")]
            for pair_prompt in pair_prompts:
                if pair_prompt not in sample_prompts_te_outputs:
                    logger.info(f"Cache Text Encoder outputs for pair prompt: {pair_prompt}")
                    tokens_and_masks = tokenize_strategy.tokenize(base_prompt) 
                    sample_prompts_te_outputs[pair_prompt] = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, text_encoder, tokens_and_masks, True
                    )

    if not conditions:
        return
    
    logger.info(f"Generating image")
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in sample_prompts:
            base_prompt = prompt_dict.get("prompt", "")
            
            prompt_has_conditions = (base_prompt in conditions or 
                                   any(key.startswith(f"{base_prompt}_pair") for key in conditions.keys()))
            
            if not prompt_has_conditions:
                continue
            

            pair_prompts = [key for key in conditions.keys() if key.startswith(f"{base_prompt}_pair")]
            
            if pair_prompts:

                for pair_idx, pair_prompt in enumerate(pair_prompts):
                    pair_prompt_dict = prompt_dict.copy()
                    pair_prompt_dict["prompt"] = pair_prompt
                    pair_prompt_dict["enum"] = pair_idx  
                    
                    sample_image_inference(
                        args,
                        accelerator,
                        flux,
                        text_encoder,
                        vae,
                        save_dir,
                        pair_prompt_dict,
                        sample_prompts_te_outputs,
                        None,
                        conditions
                    )
            else:
               
                sample_image_inference(
                    args,
                    accelerator,
                    flux,
                    text_encoder,
                    vae,
                    save_dir,
                    prompt_dict,
                    sample_prompts_te_outputs,
                    None,
                    conditions
                )

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
        args,
        accelerator: Accelerator,
        flux: flux_models.Flux,
        text_encoder,
        ae: flux_models.AutoEncoder,
        save_dir,
        prompt_dict,
        sample_prompts_te_outputs,
        prompt_replacement,
        sample_images_ae_outputs
):
    # Extract parameters from prompt_dict
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 1024) 
    height = prompt_dict.get("height", 1024) 
    scale = prompt_dict.get("scale", 1.0)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    # Ensure height and width are divisible by 16
    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"prompt: {prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # Encode prompts
    # Assuming that TokenizeStrategy and TextEncodingStrategy are compatible with Accelerator
    text_encoder_conds = []
    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        text_encoder_conds = sample_prompts_te_outputs[prompt]
        logger.info(f"Using cached text encoder outputs for prompt: {prompt}")


    if sample_images_ae_outputs and prompt in sample_images_ae_outputs:
        condition_data = sample_images_ae_outputs[prompt]
        if isinstance(condition_data, list) and len(condition_data) == 2:
            # [init_latents, final_latents]
            ae_outputs = [condition_data[0].to(accelerator.device), condition_data[1].to(accelerator.device)]
            logger.info(f"Using dual image conditions - init: {ae_outputs[0].shape}, final: {ae_outputs[1].shape}")
        else:
            # 
            ae_outputs = condition_data.to(accelerator.device)
            logger.info(f"Using single image condition: {ae_outputs.shape}")
    else:
        ae_outputs = None

    # ae_outputs = torch.load('ae_outputs.pth', map_location='cuda:0')

    # text_encoder_conds = torch.load('text_encoder_conds.pth', map_location='cuda:0')
    l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds


    logger.debug(
        f"l_pooled shape: {l_pooled.shape}, t5_out shape: {t5_out.shape}, txt_ids shape: {txt_ids.shape}, t5_attn_mask shape: {t5_attn_mask.shape}")


    weight_dtype = ae.dtype  # TODO: give dtype as argument
    packed_latent_height = height // 16
    packed_latent_width = width // 16


    logger.debug(f"packed_latent_height: {packed_latent_height}, packed_latent_width: {packed_latent_width}")


    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )

 
    timesteps = flux_train_utils.get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(
        accelerator.device, dtype=weight_dtype
    )
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    clip_l, t5xxl = text_encoder
    # ae.to("cpu")
    clip_l.to("cpu")
    t5xxl.to("cpu")

    clean_memory_on_device(accelerator.device)
    flux.to("cuda")

    for param in flux.parameters():
        param.requires_grad = False

    with accelerator.autocast(), torch.no_grad():
        x = flux_train_utils.denoise(args, flux, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps,
                                     guidance=scale, t5_attn_mask=t5_attn_mask, ae_outputs=ae_outputs)

    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    # clean_memory_on_device(accelerator.device)
    ae.to(accelerator.device)
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    ae.to("cpu")
    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict.get("enum", 0)  # Ensure 'enum' exists
    

    if "_pair" in prompt:
        pair_idx = prompt.split("_pair")[-1]
        img_filename = f"{ts_str}{seed_suffix}_pair{pair_idx}_{i}.png"

    else:
        img_filename = f"{ts_str}{seed_suffix}_{i}.png"

    
    image.save(os.path.join(save_dir, img_filename))


def setup_argparse():
    parser = argparse.ArgumentParser(description="FLUX-Controlnet-Inpainting Inference Script")

    # Paths
    parser.add_argument('--base_flux_checkpoint', type=str, required=True,
                        help='Path to BASE_FLUX_CHECKPOINT')
    parser.add_argument('--lora_weights_path', type=str, required=True,
                        help='Path to LORA_WEIGHTS_PATH')
    parser.add_argument('--clip_l_path', type=str, required=True,
                        help='Path to CLIP_L_PATH')
    parser.add_argument('--t5xxl_path', type=str, required=True,
                        help='Path to T5XXL_PATH')
    parser.add_argument('--ae_path', type=str, required=True,
                        help='Path to AE_PATH')
    parser.add_argument('--sample_images_file', type=str, required=True,
                        help='Path to SAMPLE_IMAGES_FILE')
    parser.add_argument('--sample_prompts_file', type=str, required=True,
                        help='Path to SAMPLE_PROMPTS_FILE')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save OUTPUT_DIR')
    parser.add_argument('--dataset_frame_num', type=int, choices=[4, 9], required=True,
                        help="The number of steps in the generated step diagram (choose 4 or 9)")

    parser.add_argument('--use_sliding_window', action='store_true',
                        help='Enable sliding window frame selection for grid images')
    parser.add_argument('--sliding_window_alpha', type=float, default=0.6,
                        help='Visual semantic difference weight for sliding window (default: 0.6)')
    parser.add_argument('--sliding_window_beta', type=float, default=0.2,
                        help='Text semantic difference weight for sliding window (default: 0.2)')
    parser.add_argument('--adaptive_k', action='store_true',
                        help='Enable adaptive K selection based on image differences')
    parser.add_argument('--adaptive_threshold', type=float, default=0.5,
                        help='Threshold for adaptive K selection (default: 0.5)')

    return parser.parse_args()


def main(args):
    accelerator = Accelerator(mixed_precision='bf16', device_placement=True)

    BASE_FLUX_CHECKPOINT = args.base_flux_checkpoint
    LORA_WEIGHTS_PATH = args.lora_weights_path
    CLIP_L_PATH = args.clip_l_path
    T5XXL_PATH = args.t5xxl_path
    AE_PATH = args.ae_path

    SAMPLE_IMAGES_FILE = args.sample_images_file
    SAMPLE_PROMPTS_FILE = args.sample_prompts_file
    OUTPUT_DIR = args.output_dir

    with open(SAMPLE_IMAGES_FILE, "r", encoding="utf-8") as f:
        image_lines = f.readlines()
    sample_images = [line.strip() for line in image_lines if line.strip() and not line.strip().startswith("#")]

    sample_prompts = train_util.load_prompts(SAMPLE_PROMPTS_FILE)

    # Load models onto CUDA via Accelerator
    _, [clip_l, t5xxl], ae, model = load_target_model(
        fp8_base=True,
        pretrained_model_name_or_path=BASE_FLUX_CHECKPOINT,
        disable_mmap_load_safetensors=False,
        clip_l_path=CLIP_L_PATH,
        fp8_base_unet=False,
        t5xxl_path=T5XXL_PATH,
        ae_path=AE_PATH,
        weight_dtype=torch.bfloat16,
        accelerator=accelerator
    )

    model.eval()
    clip_l.eval()
    t5xxl.eval()
    ae.eval()

    # LoRA
    multiplier = 1.0
    weights_sd = load_file(LORA_WEIGHTS_PATH)
    lora_model, _ = lora_flux.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd,
                                                          True)

    lora_model.apply_to([clip_l, t5xxl], model)
    info = lora_model.load_state_dict(weights_sd, strict=True)
    logger.info(f"Loaded LoRA weights from {LORA_WEIGHTS_PATH}: {info}")
    lora_model.eval()
    lora_model.to("cuda")

    # Set text encoders
    text_encoder = [clip_l, t5xxl]

    sample(args, accelerator, vae=ae, text_encoder=text_encoder, flux=model, output_dir=OUTPUT_DIR,
           sample_images=sample_images, sample_prompts=sample_prompts)


if __name__ == "__main__":
    args = setup_argparse()

    main(args)
