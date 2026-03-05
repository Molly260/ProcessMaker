#!/bin/bash

CKPT_PATH="./ckpt/unet/flux1-dev-fp8-e4m3fn.safetensors"
CLIP_L_PATH="./ckpt/encoder/clip_l.safetensors"
T5XXL_PATH="./ckpt/encoder/t5xxl_fp8_e4m3fn.safetensors"
AE_PATH="./ckpt/vae/ae.safetensors"

dataset_config="./dataset/config_4.toml" 
output_dir="./output/directory"
output_name="4frame"

network_dim=64
max_train_steps=50000

# Add LoRA mask
num_classes=10 

accelerate launch --config_file="./config.yaml" \
  --main_process_port=23325 train_stage1.py \
  --dataset_config $dataset_config \
  --pretrained_model_name_or_path $CKPT_PATH \
  --ae $AE_PATH \
  --clip_l $CLIP_L_PATH \
  --t5xxl $T5XXL_PATH \
  --optimizer_type came \
  --max_grad_norm 1.0 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --lr_scheduler_num_cycles 1 \
  --lr_scheduler_power 1.0 \
  --min_snr_gamma 5 \
  --output_name $output_name \
  --output_dir $output_dir \
  --network_dim $network_dim \
  --network_alpha 1.0 \
  --learning_rate 1e-4 \
  --max_train_steps $max_train_steps \
  --apply_t5_attn_mask \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --weighting_scheme logit_normal \
  --logit_mean 0 \
  --logit_std 1.0 \
  --mode_scale 1.29 \
  --timestep_sampling shift \
  --sigmoid_scale 1.0 \
  --model_prediction_type raw \
  --guidance_scale 1.0 \
  --discrete_flow_shift 3.1582 \
  --fp8_base \
  --lowram \
  --gradient_checkpointing \
  --seed 42 \
  --save_precision bf16 \
  --save_every_n_epochs 500 \
  --network_module networks.asylora_flux \
  --network_train_unet_only \
  --vae_batch_size 1 \
  --save_model_as safetensors \
  --max_data_loader_n_workers 0 \
  --mixed_precision bf16 \
  --skip_cache_check \
  --gradient_accumulation_steps 1 \
  --log_config \
  --num_classes $num_classes \
  --use_class_specific_mask \
  --use_multilayer_supervision \
  --target_supervision_layers "4,14,29" \
  --layer_supervision_weights "0.3,0.4,0.3" \
  --multilayer_supervision_weight 0.3 \
  --supervision_loss_type "cosine" \
  --supervision_schedule "staged" \
  --dynamic_layer_weights
