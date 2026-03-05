#!/bin/bash

BASE_FLUX_CHECKPOINT="./merged_4frame.safetensors"  
LORA_WEIGHTS_PATH="./output_dir/frame4_interpolation.safetensors"
OUTPUT_DIR="./output_dir"

CLIP_L_PATH="./ckpt/encoder/clip_l.safetensors"
T5XXL_PATH="./ckpt/encoder/t5xxl_fp8_e4m3fn.safetensors"  
AE_PATH="./ckpt/vae/ae.safetensors"

SAMPLE_IMAGES_FILE="./inter/images.txt"
SAMPLE_PROMPTS_FILE="./inter/prompts.txt"
dataset_frame_num=4  # 4 for 1024 or 9 for 1056 

# sliding window
USE_SLIDING_WINDOW=false  # Use sliding window or not
SLIDING_WINDOW_ALPHA=0.6  
SLIDING_WINDOW_BETA=0.2  
ADAPTIVE_K=true         
ADAPTIVE_THRESHOLD=0.065  



python inference_stage2.py \
    --base_flux_checkpoint "$BASE_FLUX_CHECKPOINT" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --clip_l_path "$CLIP_L_PATH" \
    --t5xxl_path "$T5XXL_PATH" \
    --ae_path "$AE_PATH" \
    --sample_images_file "$SAMPLE_IMAGES_FILE" \
    --sample_prompts_file "$SAMPLE_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_frame_num $dataset_frame_num \
    $(if [ "$USE_SLIDING_WINDOW" = true ]; then echo "--use_sliding_window"; fi) \
    --sliding_window_alpha "$SLIDING_WINDOW_ALPHA" \
    --sliding_window_beta "$SLIDING_WINDOW_BETA" \
    $(if [ "$ADAPTIVE_K" = true ]; then echo "--adaptive_k"; fi) \
    --adaptive_threshold "$ADAPTIVE_THRESHOLD"
