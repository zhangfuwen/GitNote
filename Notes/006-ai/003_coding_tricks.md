
## jupyter notebook
### magic commands

[参考](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

`%load filename` 将python代码加载进来，类似于bash的source。运行的效果是：
	点一次运行：代码中的文本内容会在当前cell展开显示
	再点一次运行：展开的代码会执行





## install torch

打开文档，选择用pip的安装方法，别用conda, conda容易卡住。

https://pytorch.org/get-started/locally/


### taming-transformers


```shell
git clone https://github.com/CompVis/taming-transformers
cd taming-transformers

conda env create -f environment.yaml
conda activate taming
pip install -e .

```

### conda install cudatoolkit

```python
import json

data = {
	"name": "dean",
	"age":30
}
print(json.dumps(data))
```

```js
console.log("hello")
```

https://anaconda.org/nvidia/cuda-nvcc

```bash
conda install nvidia::cuda-toolkit  
conda install nvidia/label/cuda-11.3.0::cuda-toolkit  
conda install nvidia/label/cuda-11.3.1::cuda-toolkit  
conda install nvidia/label/cuda-11.4.0::cuda-toolkit  
conda install nvidia/label/cuda-11.4.1::cuda-toolkit  
conda install nvidia/label/cuda-11.4.2::cuda-toolkit  
conda install nvidia/label/cuda-11.4.3::cuda-toolkit  
conda install nvidia/label/cuda-11.4.4::cuda-toolkit  
conda install nvidia/label/cuda-11.5.0::cuda-toolkit  
conda install nvidia/label/cuda-11.5.1::cuda-toolkit  
conda install nvidia/label/cuda-11.5.2::cuda-toolkit  
conda install nvidia/label/cuda-11.6.0::cuda-toolkit  
conda install nvidia/label/cuda-11.6.1::cuda-toolkit  
conda install nvidia/label/cuda-11.6.2::cuda-toolkit  
conda install nvidia/label/cuda-11.7.0::cuda-toolkit  
conda install nvidia/label/cuda-11.7.1::cuda-toolkit  
conda install nvidia/label/cuda-11.8.0::cuda-toolkit  
conda install nvidia/label/cuda-12.0.0::cuda-toolkit  
conda install nvidia/label/cuda-12.0.1::cuda-toolkit  
conda install nvidia/label/cuda-12.1.0::cuda-toolkit  
conda install nvidia/label/cuda-12.1.1::cuda-toolkit  
conda install nvidia/label/cuda-12.2.0::cuda-toolkit  
conda install nvidia/label/cuda-12.2.1::cuda-toolkit  
conda install nvidia/label/cuda-12.2.2::cuda-toolkit  
conda install nvidia/label/cuda-12.3.0::cuda-toolkit  
conda install nvidia/label/cuda-12.3.1::cuda-toolkit  
conda install nvidia/label/cuda-12.3.2::cuda-toolkit  
conda install nvidia/label/cuda-12.4.0::cuda-toolkit  
conda install nvidia/label/cuda-12.4.1::cuda-toolkit  
conda install nvidia/label/cuda-12.5.0::cuda-toolkit  
conda install nvidia/label/cuda-12.5.1::cuda-toolkit
```

### conda install nvcc

```bash
conda install nvidia::cuda-nvcc  
conda install nvidia/label/cuda-11.3.0::cuda-nvcc  
conda install nvidia/label/cuda-11.3.1::cuda-nvcc  
conda install nvidia/label/cuda-11.4.0::cuda-nvcc  
conda install nvidia/label/cuda-11.4.1::cuda-nvcc  
conda install nvidia/label/cuda-11.4.2::cuda-nvcc  
conda install nvidia/label/cuda-11.4.3::cuda-nvcc  
conda install nvidia/label/cuda-11.4.4::cuda-nvcc  
conda install nvidia/label/cuda-11.5.0::cuda-nvcc  
conda install nvidia/label/cuda-11.5.1::cuda-nvcc  
conda install nvidia/label/cuda-11.5.2::cuda-nvcc  
conda install nvidia/label/cuda-11.6.0::cuda-nvcc  
conda install nvidia/label/cuda-11.6.1::cuda-nvcc  
conda install nvidia/label/cuda-11.6.2::cuda-nvcc  
conda install nvidia/label/cuda-11.7.0::cuda-nvcc  
conda install nvidia/label/cuda-11.7.1::cuda-nvcc  
conda install nvidia/label/cuda-11.8.0::cuda-nvcc  
conda install nvidia/label/cuda-12.0.0::cuda-nvcc  
conda install nvidia/label/cuda-12.0.1::cuda-nvcc  
conda install nvidia/label/cuda-12.1.0::cuda-nvcc  
conda install nvidia/label/cuda-12.1.1::cuda-nvcc  
conda install nvidia/label/cuda-12.2.0::cuda-nvcc  
conda install nvidia/label/cuda-12.2.1::cuda-nvcc  
conda install nvidia/label/cuda-12.2.2::cuda-nvcc  
conda install nvidia/label/cuda-12.3.0::cuda-nvcc  
conda install nvidia/label/cuda-12.3.1::cuda-nvcc  
conda install nvidia/label/cuda-12.3.2::cuda-nvcc  
conda install nvidia/label/cuda-12.4.0::cuda-nvcc  
conda install nvidia/label/cuda-12.4.1::cuda-nvcc  
conda install nvidia/label/cuda-12.5.0::cuda-nvcc  
conda install nvidia/label/cuda-12.5.1::cuda-nvcc
```

# conda environment


## cuda build and execute

install cuda tool kit for compilation.

```bash
conda create -n myenv
conda install -c nvidia cuda cuda-nvcc
```

example code(test.cu):

```cuda
#include <stdio.h>

// CUDA kernel to perform vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size = 1024; // Size of the vectors
    int a[size], b[size], c[size]; // Input and output arrays

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Declare GPU memory pointers
    int *dev_a, *dev_b, *dev_c;
    // Allocate GPU memory
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // Copy input arrays from host to GPU memory
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch CUDA kernel on the GPU
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);

    // Copy the result back from GPU to host memory
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < size; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}


```

## build and run

```bash
nvcc test.cu -o example
./example
```

# conda install pytorch and pytorch-cuda


## nvidia version matching

nvidia kernel version, toolkit version must roughly match.

to see kernel version, use `nvidia-smi` command:

```bash
nvidia-smi     
Tue Apr 30 18:06:30 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080        Off | 00000000:01:00.0  On |                  N/A |
| 32%   45C    P8              12W / 180W |    289MiB /  8192MiB |      9%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

```

` CUDA Version: 12.2  ` is what we want to know.

to figure out cudatoolkit version, use `nvcc --version` command:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

```

`cuda_12.4.r12.4` is what we want to know.


## install pytorch 

With `pytorch` installed, you can run your model on `cpu`.
With `pytorch-cuda` installed, you can run your model on `cuda`.

Install pytorch:

```bash 
conda install pytorch=2.2
```
or

```bash
conda install pytorch==2.2.2
```

Single equal sign indicates none exact match, while double equal sign means exact match.



To test if cuda is available:

```
python3
>>> import torch 
>>> print(torch.cuda.is_available())
False
```

Install pytorch-cuda:

```bash
conda install pytorch-cuda=12.4 -c pytorch
```

The version better matches the cuda version you got using `nvidia-smi` command.

Sometimes, the version you wanted does not exist, try some other versions as long as the major version matches:

```
conda search pytorch-cuda -c pytorch
Loading channels: done
# Name                       Version           Build  Channel             
pytorch-cuda                    11.6      h867d48c_0  pytorch             
pytorch-cuda                    11.6      h867d48c_1  pytorch             
pytorch-cuda                    11.7      h67b0de4_0  pytorch             
pytorch-cuda                    11.7      h67b0de4_1  pytorch             
pytorch-cuda                    11.7      h67b0de4_2  pytorch             
pytorch-cuda                    11.7      h778d358_3  pytorch             
pytorch-cuda                    11.7      h778d358_5  pytorch             
pytorch-cuda                    11.8      h7e8668a_3  pytorch             
pytorch-cuda                    11.8      h7e8668a_5  pytorch             
pytorch-cuda                    11.8      h8dd9ede_2  pytorch             
pytorch-cuda                    12.1      ha16c6d3_5  pytorch             
pytorch-cuda                    12.4      hc786d27_6  pytorch 
```

For example, if your cuda version is 11.3, you may try 11.7 which has backward compatibility with version 11.3.

Sometimes, pytorch-cuda would have dependency conflicts with python version. so you may downgrade to an older version of python as well:

```bash
conda install python=3.8 pytorch-cuda=12.4  -c pytorch -v
```

better follow instructions here: https://pytorch.org/get-started/previous-versions/


## vllm

### 安装

```bash
pip install vllm
```

### 运行

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

### 参数调节

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

### 流式输出

```
from vllm import LLM, SamplingParams
import time

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
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





prompt = "详细介绍一下北京"

engine_args = AsyncEngineArgs(model=path, trust_remote_code=True, dtype="float16",
    max_model_len=4096)
engine = LLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER)

#help(engine)

#results_generator = iterate_with_cancellation(
#        results_generator, is_cancelled=False)

from transformers import AutoTokenizer
from vllm.inputs import (TextPrompt,
                         TokensPrompt)
tokenizer = AutoTokenizer.from_pretrained(path)



def xx(prompts):
    
    for prompt in prompts:
        history=[]
        history.append({"role": "user", "content": prompt})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors='pt')
        ids = inputs["input_ids"].cpu()[0].tolist()
        tp = TokensPrompt(prompt_token_ids = ids)
        print(tp)
        print(history_str)

        start = time.time()
        request_id = random_uuid()
        engine.add_request(str(request_id), tp, sampling_params)

    count = 0
    while engine.has_unfinished_requests():
        responses = engine.step()
        count += 1
        if count > 5:
            break
        
        print(responses)
        for resp in responses:
            print(f"req_id:{resp.request_id}, text:{resp.outputs[0].text}, len:{len(resp.outputs[0].token_ids)}")
        
        print("----------------")
    

    #async for token in results_generator:
    #    count+=1
    #    end = time.time()
    #    if count == 1 or token.finished :
    #        data = {
    #            "text": token.outputs[0].text,
    #            "token_ids":token.outputs[0].token_ids,
    #        }
    #        print(f'{data["text"][0:5]}, token_num:{len(data["token_ids"])}, time: {end-start}', end='', flush=True)
    #        yield data
            


# Using the sync generator in a for loop
#for item in sync_generator(xx(prompt)):
#    print(item["text"], len(item["token_ids"]))

xx([prompt, "你好呀"])
        
        
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

## other tricks

### 无Pytorch情况下只使用tokenizer

```bash
pip3 install transformers

```

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
model_inputs = tokenizer(["hello, I am dean"])
print(model_inputs)

```

显示结果：

```json
{'input_ids': [[14990, 11, 358, 1079, 72862]], 'attention_mask': [[1, 1, 1, 1, 1]]}
```

### nvidia-cuda-pytorch trouble-shooting

```

Traceback (most recent call last):
  File "./qwen_run_rich_text_pipeline.py", line 156, in <module>
    extracted_formats = model_chat(extract_format_model, format_inputs, extract_format_system_prompt)
  File "./qwen_run_rich_text_pipeline.py", line 72, in model_chat
    input_ids.to(device),
  File "/root/anaconda3/envs/qwen/lib/python3.8/site-packages/torch/cuda/__init__.py", line 293, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 803: system has unsupported display driver / cuda driver combination
```
解决办法：

```bash
echo $CUDA_HOME 
# /usr/local/cuda
which nvcc
# /root/anaconda3/envs/qwen/bin/nvcc

export CUDA_HOME="/roo/anaconda3/envs/qwen/"

export LD_LIBRARY_PATH=/opt/tiger/native_libhdfs/lib/native:/opt/tiger/jdk/jdk8u265-b01/jre/lib/amd64/server:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop/lib/native/ufs:/opt/tiger/native_libhdfs/lib/native:/opt/tiger/jdk/jdk8u265-b01/jre/lib/amd64/server:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop/lib/native/ufs:/opt/tiger/yarn_deploy/hadoop/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lib/native:/opt/tiger/yarn_deploy/hadoop_current/lzo/lib:/root/anaconda3/envs/qwen//lib 
```

LD_LIBRARY_PATH关系非常大，里面文件夹的顺序也很重要。
特别是当你的系统里有多种方式安装的cuda runtime的时候，比如你linux的方式安装了一些，用conda的方式又在venv里安装了一些。这时最好只有一个发挥作用。