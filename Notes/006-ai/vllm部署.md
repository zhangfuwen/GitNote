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

# 流式输出

```python
from vllm import LLM, SamplingParams
import time

import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (random_uuid)

#path="/mnt/bn/znzx-public/models/gemma-2-2b-it"
#llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")  # Name or path of your model
#help(LLM)
path="/mnt/bn/znzx-public/models/MiniCPM-2B-dpo-bf16"

#help(LLM)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)


request_id = random_uuid()


prompt = "详细介绍一下北京"

engine_args = AsyncEngineArgs(model=path, trust_remote_code=True, dtype="float16",
    max_model_len=4096)
engine = AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER)

#results_generator = iterate_with_cancellation(
#        results_generator, is_cancelled=False)

async def xx(prompt):
    start = time.time()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    count = 0
    async for token in results_generator:
        count+=1
        end = time.time()
        if count == 1 or token.finished :
            data = {
                "text": token.text,
                "token_ids":token.token_ids,
            }
            print(f"{data}, {end-start}", end='', flush=True)
            yield data
            
def sync_generator_from_async(aiter):
    loop = asyncio.get_event_loop()
    async def inner():
        async for item in aiter:
            yield item
    return inner()

asyncio.run(xx(prompt))
        
        
#llm = LLM(model=path, trust_remote_code=True, dtype="float16",
#    max_model_len=4096)  # Name or path of your model
#start = time.time()
#output = llm.generate("详细介绍一下北京", sampling_params)
#end = time.time()
#duration = end - start
#print(output[0])
#out_tokens = len(output[0].outputs[0].token_ids)
#print(f"output_tokens:{out_tokens}, time:{duration}, output speed: {out_tokens/duration}")

#start = time.time()
#output = llm.generate("详细介绍一下北京", sampling_params)
#end = time.time()
#duration = end - start
#print(output[0])
#out_tokens = len(output[0].outputs[0].token_ids)
#print(f"output_tokens:{out_tokens}, time:{duration}, output speed: {out_tokens/duration}")


```