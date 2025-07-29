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

| **Base Model**                              | **Dataset** | **Defense Solutions** | **Bash Scripts**                                     |
|----------------------------------------------|-------------|-----------------------------------------|------------------------------------------------------|
| **Llama-3.2-3B**                  | **Alpaca**  | StruQ                                   | `./scripts/llama3b/alpaca/struq_alpaca.sh`             |
|                                              |             | ISE                                     | `./scripts/llama3b/alpaca/ise_alpaca.sh`               |
|                                              |             | PosSep                                  | `./scripts/llama3b/alpaca/possep_alpaca.sh`            |
|                                              |             | InstFuse                                | `./scripts/llama3b/alpaca/instfuse_alpaca.sh`          |
| **Llama-3.2-3B**                  | **SEP**     | *StruQ*                                 | `./scripts/llama3b/sep/struq_alpaca.sh`                |
|                                              |             | *ISE*                                   | `./scripts/llama3b/sep/ise_alpaca.sh`                  |
|                                              |             | *PosSep*                                | `./scripts/llama3b/sep/possep_alpaca.sh`               |
|                                              |             | *InstFuse*                              | `./scripts/llama3b/sep/instfuse_alpaca.sh`             |
| **Ministral-3b-instruct**          | **Alpaca**  | *StruQ*                                 | `./scripts/ministral3b/alpaca/struq_alpaca.sh`         |
|                                              |             | *ISE*                                   | `./scripts/ministral3b/alpaca/ise_alpaca.sh`           |
|                                              |             | *PosSep*                                | `./scripts/ministral3b/alpaca/possep_alpaca.sh`        |
|                                              |             | *InstFuse*                              | `./scripts/ministral3b/alpaca/instfuse_alpaca.sh`      |
| **Ministral-3b-instruct**          | **SEP**     | *StruQ*                                 | `./scripts/ministral3b/sep/struq_alpaca.sh`            |
|                                              |             | *ISE*                                   | `./scripts/ministral3b/sep/ise_alpaca.sh`              |
|                                              |             | *PosSep*                                | `./scripts/ministral3b/sep/possep_alpaca.sh`           |
|                                              |             | *InstFuse*                              | `./scripts/ministral3b/sep/instfuse_alpaca.sh`         |


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