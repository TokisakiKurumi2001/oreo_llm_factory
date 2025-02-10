# Basic toolbox for Llama-factory

Future repo based on Llama-factory can rely on this repo to further develop.

```bash
git clone https://github.com/TokisakiKurumi2001/base-llm-factory /path/to/your/new/project
cd /path/to/your/new/project
rm -rf .git
git init
git add .
git config user.name "<your_username>"
git config user.email "<your_email>"
git commit -m "Init repo"
git branch -M main
git remote add origin <link_to_your_git>
git push -u origin main
```

## Some basic stuff this repo has tested

### Pre-training

Llama-3.2-3B (1024 sequence length) full fine-tuning on Single GPU with Zero-3 offloading optimizer and parameters (256GB CPU RAM + 32 CPU Core + 40GB VRAM GPU)

```bash
llamafactory-cli train training_script/basic/train/pt.yaml
```

### Supervised Fine-tuning

Use LoRA.

```bash
llamafactory-cli train training_script/basic/train/sft.yaml
```

### Reward modelling

```bash
llamafactory-cli train training_script/basic/train/reward.yaml
```

### DPO

```bash
llamafactory-cli train training_script/basic/train/dpo.yaml
```

### KTO

```bash
llamafactory-cli train training_script/basic/train/kto.yaml
```