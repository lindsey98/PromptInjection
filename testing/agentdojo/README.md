# AgentDojo Evaluation

Evaluate DRIP (and baselines) on the [AgentDojo](https://github.com/ethz-spylab/agentdojo) agentic prompt-injection benchmark.

## Install

```bash
pip install agentdojo==0.1.35
```

## How to run

Each command below runs the **no-attack** setting. To run **with an attack**, append:

```bash
--attack [important_instructions|ignore_previous]
```

The undefended and Meta SecAlign models are served through a local vLLM server, so start that first (the command is shown in each block). DRIP loads the model directly and needs no server.

### Undefended model

```bash
bash run_local_vlm.sh   # start local vLLM server

python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/llama8b
```

### Meta SecAlign model

Adds `--tool-delimiter input`.

```bash
bash run_local_vlm_metasecalign.sh   # start local vLLM server

python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/metasecalign8b \
  --tool-delimiter input
```

### DRIP

```bash
python -m testing.agentdojo.run_agentdojo \
  --mode fuse \
  --model_name_or_path [model_path] \
  --customized_model_class LlamaForCausalLMDRIP \
  --logdir ./agentdojo_runs/llama8b_drip
```
