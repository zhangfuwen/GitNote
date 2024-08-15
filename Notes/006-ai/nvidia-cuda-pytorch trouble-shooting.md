## CUDA_HOME
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