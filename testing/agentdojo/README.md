

### DRIP

```bash
python run_agentdojo.py \
  --mode fuse \
  -m /path/to/Llama-3-8B-fuse-dpo \
  --customized_model_class LlamaForCausalLMDRIP \
  --attack important_instructions \
  --suites banking
```

### Normal models

```bash
# Terminal 1
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --dtype auto \
  --tensor-parallel-size 2 \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
  
# Terminal 2
python run_agentdojo.py \
  --mode official \
  -m local \
  --model-name Qwen2.5-32B-Instruct \
  --attack important_instructions \
  --defense repeat_user_prompt \
  --suites banking \
  --logdir ./agentdojo_runs/qwen3_30b
```