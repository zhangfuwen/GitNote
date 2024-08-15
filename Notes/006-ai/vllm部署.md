# 安装

```bash
pip install vllm
```

# 运行
```python
from vllm import LLM, SamplingParams
import time

#llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")  # Name or path of your model
#help(LLM)
path="./MiniCPM-2B-dpo-bf16"
llm = LLM(model=path, trust_remote_code=True,
    max_model_len=4096)  # Name or path of your model
help(LLM)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)

start = time.time()
output = llm.generate("详细介绍一下北京", sampling_params)
end = time.time()
duration = end - start
#print(output[0])
out_tokens = len(output[0].outputs[0].token_ids)
print(f"output_tokens:{out_tokens}, time:{duration}, output speed: {out_tokens/duration}")

start = time.time()
output = llm.generate("详细介绍一下北京", sampling_params)
end = time.time()
duration = end - start
#print(output[0])
out_tokens = len(output[0].outputs[0].token_ids)
print(f"output_tokens:{out_tokens}, time:{duration}, output speed: {out_tokens/duration}")
```

# 参数调节

显存占用：
```python
llm2 = LLM(model="./MiniCPM-2B-128k", trust_remote_code=True,
        gpu_memory_utilization=0.2,
    max_model_len=8192)
```
0.2表示总共使用20%的显存。不设置情况下vllm会占用所有显存，用于扩充能支持的batch数。

nVidia显存占用量获取：
```python
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
```

