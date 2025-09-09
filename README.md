# Introduction


# Setup

Install requirements

```bash
conda create --name prompt python=3.10
conda activate prompt
conda install pip
pip install -r requirements.txt
```

# Training

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/struq_sep.sh
```

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/secalign_sep.sh
```

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/ise_sep.sh
```

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/possep_sep.sh
```

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/instfuse_sep_origdata.sh
```

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama8b/sep/instfuse_sep_newdata.sh
```

# Evaluation

When prompted, please specify the CUDA device ID (i.e. a single number) you want to run the model with.
And also point to your saved model path.

## Natural language injection attack

```bash
./scripts/nl_injection1.sh
```

```bash
./scripts/nl_injection2.sh
```

```bash
./scripts/nl_injection3.sh
```

## Optimization-based attack

### GCG

```bash
./scripts/gcg_injection.sh
```

### AdvPrompter

```bash
./scripts/advprompter_injection.sh
```

## SEP benchmark attack

```bash
./scripts/sep.sh
```