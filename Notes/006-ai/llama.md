---

title: llama

---

# PC上运行

## 程序编译

```bash

sudo apt install git ninja-build cmake

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
cmake -B build -G Ninja
ninja -C build
cd -
```

## 模型权重文件获取

获取llama模型权重文件：

```bash
if ! command -v pip3; then
  sudo apt install python3 python3-pip
if
pip install transformers # 这个是pyllama的运行时依赖，pyllama可能没写。不安装这个会造成运行时失败
pip install pyllama -U
python -m llama.download --model_size 7B

```
model_size可选值：7B,13B,30B,65B。其中7B模型大概14GB以内，13B模型26GB左右，依些类推，乘以2就行了。
运行时需要把模型完全加载到内存或显存才能工作。
可以通过量化的方式把模型降到原来的四分之一大小。

下载完的目录结构：
✅ pyllama_data/tokenizer.model
✅ pyllama_data/tokenizer_checklist.chk
✅ pyllama_data/7B/consolidated.00.pth
✅ pyllama_data/7B/params.json
✅ pyllama_data/7B/checklist.chk

模型权重文件是有格式的，这种方式下载到的是pytorch格式的，需要传换成ggml格式才能用：

## 格式转换

```bash
python3 ./llama.cpp/convert-pth-to-ggml.py ./pyllama_data/7B 1
```

## 量化

成生的文件是16位浮点数的，整体size比较大，可以量化为int4的：

```bash
./llamma.cpp/build/bin/quantize ./pyllama_data/7B/ggml-model-f16.bin ./pyllama_data/7B/ggml-model-q4_0.bin 2
```

## 运行

```bash
./llama.cpp/build/bin/main -m ./pyllama_data/7B/ggml-model-q4_0.bin -t 4 -n 128 --color -i -r "User:"
```

# 扩展的中文友好的模型

## Linly

这个git repo上分享是一个tencent pretrain格式的模型：
https://huggingface.co/Linly-AI/ChatFlow-7B

转换为llama格式后可以直接用。

### 下载

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/ChatFlow-7B
git clone https://github.com/ProjectD-AI/llama_inference
```

### 格式转换

从tencent pretrain转换为pytorch格式：

```bash
https://github.com/Tencent/TencentPretrain

mv pyllama_data/7B/consolidated.00.pth{,.back-original} # 备分，将新生成的文件放到pyllama文件夹下

python3 TencentPretrain/scripts/convert_tencentpretrain_to_llama.py --input_model_path ChatFlow-7B/chatflow_7b.bin \
                                                    --output_model_path pyllama_data/7B/consolidated.00.pth \
                                                    --layers 32

```

后续使用需要重新经历[格式转换](#格式转换)、[量化](#量化)、[运行](#运行) 三个步骤。

# 其他运行方式

## llama.cpp其他方式运行

llama.cpp可以使用CLBlast或cuBlas做为后端运行。使用cuBlas可以直接运行在GPU上。

cuBlas:

```bash

cd llama.cpp
rm -rf build
cmake -B build -G Ninja -DLLAMA_CUBLAS=ON
ninja -C build
cd -

```

clBlast:

```bash

cd llama.cpp
rm -rf build
cmake -B build -G Ninja -DLLAMA_CLBLAST=ON # -DCLBlast_dir=/some/path
ninja -C build
cd -

```

运行时需要加上`--gpu-layers`参数。

opencl的话，可以用环境变量指定设备和平台

```bash

GGML_OPENCL_PLATFORM=1 ./main ...
GGML_OPENCL_DEVICE=2 ./main ...
GGML_OPENCL_PLATFORM=Intel ./main ...
GGML_OPENCL_PLATFORM=AMD GGML_OPENCL_DEVICE=1 ./main ...
```

## NDK 编译运行

```bash
cmake -B build -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=29 -DCMAKE_TOOLCHAIN_FILE=/home/xxx/android-ndk-r25c/build/cmake/android.toolchain.cmake  -G Ninja
ninja -C build

```