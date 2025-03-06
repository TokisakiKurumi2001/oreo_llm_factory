wandb offline
export CUDA_VISIBLE_DEVICES=0
huggingface-cli download jwhj/OREO-Qwen2.5-Math-1.5B-Train --repo-type dataset
huggingface-cli download Qwen/Qwen2.5-Math-1.5B
llamafactory-cli train training_script/oreo/train/sft.yaml