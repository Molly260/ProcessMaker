#!/bin/bash

CKPT_PATH="./ckpt/unet/flux1-dev-fp8-e4m3fn.safetensors
CLIP_L_PATH="./ckpt/encoder/clip_l.safetensors"
T5XXL_PATH="./ckpt/encoder/t5xxl_fp16.safetensors"
AE_PATH="./ckpt/vae/ae.safetensors"
LORA_PATH="./output/directory/4frame.safetensors"
OUTPUT_DIR="./output"


PROMPT=""
CLASS_ID=18


python flux_minimal_inference_asylora.py --ckpt_path $CKPT_PATH \
  --clip_l $CLIP_L_PATH \
  --t5xxl $T5XXL_PATH \
  --ae $AE_PATH \
  --prompt "$PROMPT" \
  --width 1024 \
  --height 1024 \
  --steps 25 \
  --dtype bf16 \
  --output_dir $OUTPUT_DIR \
  --flux_dtype fp8 \
  --offload \
  --lora_weights $LORA_PATH \
  --class_id $CLASS_ID
