export VLLM_WORKER_MULTIPROC_METHOD=spawn 
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 
# export VLLM_USE_MODELSCOPE=True 
# CUDA_VISIBLE_DEVICES=0,1,2,3


vllm serve "models/Qwen-3-32B" --load-format "safetensors" --port 8000 \
--served-model-name "Qwen-3-32B" \
--max-model-len 32768 \
--tensor-parallel-size 4 \
--trust-remote-code \
--gpu_memory_utilization 0.9 
