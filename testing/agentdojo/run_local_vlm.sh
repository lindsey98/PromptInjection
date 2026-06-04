export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export VLLM_USE_COMPILE=0
export OMP_NUM_THREADS=16
export NCCL_SOCKET_IFNAME=lo,eth0
export VLLM_NO_USAGE_STATS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export no_proxy="localhost,127.0.0.1,0.0.0.0"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --served-model-name local \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --max-model-len 32768 \
  --kv-cache-dtype auto