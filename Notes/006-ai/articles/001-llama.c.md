# 1000行代码，读懂大模型推理：手把手解析llama.c，这才是学习LLM的终极入门课


你是否曾被GPT-4、Llama 3、Qwen这些“庞然大物”吓退？  
你是否翻遍了Hugging Face的源码，却在几千行的PyTorch和CUDA中迷失方向？  
你是否渴望真正“看懂”一个大语言模型是如何运行的——不是调API，而是从零构建它的推理逻辑？

今天，我要向你推荐一个“反常识”的宝藏项目：**llama.c**。

它只有**1000行C代码**。  
没有PyTorch，没有CUDA，没有依赖库。  
没有复杂的分布式训练，没有量化优化，没有TensorRT。  
但它，**完整实现了Llama 2的推理核心**。

它不追求性能，只追求**清晰**。  
它不为生产，只为**理解**。

而它的作者，是一位名叫**Andrej Karpathy**的AI研究员——特斯拉前AI总监、斯坦福博士、《神经网络与深度学习》课程创始人，也是你可能在YouTube上见过的那位“用Python画神经网络”的极客。

---

### 为什么说llama.c是学习LLM的“黄金入门”？

在AI圈，我们总被“大模型”“多卡训练”“参数量万亿”这些词包围。但真相是：**一个Transformer模型的推理，本质上是一个非常干净的数学流程**。

llama.c的诞生，就是为了撕开这层“神秘感”。

> “如果你不能用1000行代码写出来，说明你还没真正理解它。”  
> —— Andrej Karpathy

他用C语言，从头实现Llama 2的前向传播，包括：

- Token Embedding
- Layer Normalization
- Multi-Head Attention（含RoPE位置编码）
- Feed-Forward Network
- Softmax + 输出投影

所有代码，**没有一行是黑箱**。  
所有矩阵运算，**手动展开**。  
所有张量维度，**清晰标注**。  
所有注释，**像老师讲课一样细致**。

你不需要GPU，不需要Python环境，甚至不需要编译器——用gcc就能跑起来。  
你只需要一颗愿意思考的心。

---

### 代码逻辑总览：从Token到Text，1000行走完Llama 2推理全流程

llama.c的核心流程，可以概括为以下5步：

1. **加载模型**：读取从Hugging Face导出的二进制权重文件（.bin）  
2. **Tokenize输入**：用内置的Byte-Pair Encoding（BPE）词表，把文字转成数字序列  
3. **前向传播**：逐层执行Attention + FFN，传递隐藏状态  
4. **采样输出**：用Temperature + Top-p采样，从概率分布中生成下一个Token  
5. **循环生成**：直到生成结束符或达到最大长度，输出完整文本

整个系统，**没有框架依赖，没有动态图，没有自动微分**。  
你看到的，就是模型在“思考”时，每一步发生了什么。

---

### 逐行解析：llama.c核心代码拆解

下面，我们从主函数开始，一段一段带你读懂这1000行“神作”。

---

#### 📌 1. 主函数入口：`main()` —— 一切从这里开始

```c
int main(int argc, char *argv[]) {
    // 1. 加载模型权重
    Transformer transformer;
    load_model("weights/llama-2-7b.bin", &transformer);

    // 2. 初始化词表
    Tokenizer tokenizer;
    load_tokenizer("weights/tokenizer.bin", &tokenizer);

    // 3. 输入提示词
    char *prompt = "The capital of France is ";
    
    // 4. 编码输入
    int *tokens = encode(tokenizer, prompt, &num_tokens);

    // 5. 开始生成
    generate(transformer, tokenizer, tokens, num_tokens, 100);

    return 0;
}
```

**解读**：  
这就是整个系统的“指挥中心”。没有TensorFlow，没有Jupyter，只有清晰的函数调用链。  
你一眼就能看出：**模型加载 → 输入编码 → 生成输出**。  
没有隐藏逻辑，没有魔法函数。  
**这就是工程的美。**

---

#### 📌 2. 模型加载：`load_model()` —— 权重从文件到内存

```c
void load_model(char *filename, Transformer *model) {
    FILE *f = fopen(filename, "rb");
    fread(&model->config, sizeof(Config), 1, f); // 读取模型配置
    model->wte = malloc(model->config.vocab_size * model->config.dim * sizeof(float)); // 词嵌入
    fread(model->wte, sizeof(float), model->config.vocab_size * model->config.dim, f);
    // ... 逐层加载 attention、ffn、norm 权重
    fclose(f);
}
```

**解读**：  
`Config`结构体里，记录了模型的层数、头数、维度等超参数。  
所有权重（词嵌入、注意力QKV、前馈网络权重）被**连续读入内存**，没有分层封装。  
你看到的不是`model.transformer.h[0].attn.wq.weight`，而是`model.wq[0]`——**直接索引，毫无抽象**。  
这正是学习的精髓：**剥离框架，直面数据**。

---

#### 📌 3. Tokenizer：`encode()` —— 文字如何变成数字？

```c
int* encode(Tokenizer t, char *text, int *out_len) {
    int len = strlen(text);
    int *tokens = malloc(MAX_SEQ_LEN * sizeof(int));
    *out_len = 0;

    for (int i = 0; i < len; ) {
        int max_match = 0;
        int best_idx = 0;
        for (int j = 0; j < t.vocab_size; j++) {
            if (strncmp(text + i, t.vocab[j], strlen(t.vocab[j])) == 0 &&
                strlen(t.vocab[j]) > max_match) {
                max_match = strlen(t.vocab[j]);
                best_idx = j;
            }
        }
        tokens[(*out_len)++] = best_idx;
        i += max_match;
    }
    return tokens;
}
```

**解读**：  
这是最原始的BPE编码实现。  
它不依赖Python的`transformers`库，而是**暴力遍历词表**，找最长匹配。  
你甚至可以打印出`t.vocab[0]`，看到`"Ġ"`（空格的特殊编码）、`"the"`、`"ing"`这些子词单元。  
**你终于明白：原来“AI理解文字”，是从“拆字”开始的。**

---

#### 📌 4. 前向传播核心：`transformer_forward()` —— Attention的真面目

```c
void transformer_forward(Transformer *model, int *tokens, int n_tokens, float *logits) {
    // Step 1: Embedding
    float *x = model->x; // 当前隐藏状态
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < model->config.dim; j++) {
            x[i * model->config.dim + j] = model->wte[tokens[i] * model->config.dim + j];
        }
    }

    // Step 2: Layer-by-layer Transformer blocks
    for (int l = 0; l < model->config.n_layers; l++) {
        // RMSNorm
        rmsnorm(x, model->rms_att_weight[l], model->config.dim, n_tokens);

        // Multi-Head Attention
        attention(model, x, l, n_tokens);

        // Add residual
        for (int i = 0; i < n_tokens * model->config.dim; i++) {
            x[i] += model->residual[i];
        }

        // RMSNorm again
        rmsnorm(x, model->rms_ffn_weight[l], model->config.dim, n_tokens);

        // Feed Forward Network
        ffn(model, x, l, n_tokens);

        // Add residual again
        for (int i = 0; i < n_tokens * model->config.dim; i++) {
            x[i] += model->residual[i];
        }
    }

    // Final RMSNorm
    rmsnorm(x, model->rms_final_weight, model->config.dim, n_tokens);

    // Final projection to vocab
    matmul(x, model->wcls, logits, n_tokens, model->config.dim, model->config.vocab_size);
}
```

**这是整段代码的灵魂！**

我们拆解一下：

- **Embedding层**：用`tokens[i]`作为索引，查表得到词向量。  
- **RMSNorm**：比LayerNorm更简单，只做均方根归一化，无偏置。  
- **Attention**：  
  - 计算Q/K/V（`q = x @ wq`, `k = x @ wk`, `v = x @ wv`）  
  - 应用**RoPE**（旋转位置编码）——这是Llama的关键创新，用复数旋转代替位置Embedding  
  - 计算Attention Score：`scores = q @ k.T / sqrt(d)`  
  - Softmax + dropout（这里省略）  
  - 加权求和：`output = scores @ v`  
  - 拼接多头，投影回原维度  
- **FFN**：`x → W1 → GELU → W2 → output`，标准两层MLP  
- **残差连接**：`x = x + attention_output`，每一层都加回来

**你看到的不是“Transformer模块”，而是每一行矩阵乘法、每一个循环、每一个维度对齐。**

---

#### 📌 5. RoPE位置编码：Llama的优雅设计

```c
void apply_rope(float *x, int head_dim, int pos, int n_heads, int seq_len) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / head_dim);
            float theta = pos * freq;
            float cos_val = cosf(theta);
            float sin_val = sinf(theta);

            int idx1 = h * head_dim + i;
            int idx2 = h * head_dim + i + head_dim / 2;

            float x1 = x[idx1];
            float x2 = x[idx2];
            x[idx1] = x1 * cos_val - x2 * sin_val;
            x[idx2] = x1 * sin_val + x2 * cos_val;
        }
    }
}
```

**解读**：  
这是Llama 2最惊艳的设计之一。  
传统Transformer用固定的Positional Embedding，而RoPE**把位置信息编码进向量的旋转角度**。  
这段代码，用三角函数，把每个词向量的前半部分和后半部分进行**旋转**，从而“记住”它在序列中的位置。  
**数学之美，尽在其中。**

---

#### 📌 6. 采样生成：`generate()` —— AI如何“决定”下一个词？

```c
void generate(Transformer *model, Tokenizer tokenizer, int *tokens, int n_tokens, int max_new_tokens) {
    for (int t = 0; t < max_new_tokens; t++) {
        float *logits = malloc(model->config.vocab_size * sizeof(float));
        transformer_forward(model, tokens, n_tokens, logits);

        // Temperature sampling
        float temperature = 0.8f;
        float probs[model->config.vocab_size];
        softmax(logits, probs, model->config.vocab_size, temperature);

        // Top-p sampling
        int next_token = sample_top_p(probs, model->config.vocab_size, 0.9f);

        // Add to sequence
        tokens[n_tokens++] = next_token;

        // Print token
        char *word = decode(tokenizer, next_token);
        printf("%s", word);

        // Stop if end-of-sequence
        if (next_token == 2) break; // EOS token
        free(logits);
    }
}
```

**解读**：  
AI不是“猜”词，是**按概率抽样**。  
- `softmax` 把logits转成概率分布  
- `temperature=0.8`：让分布更“平滑”，避免过于保守  
- `top_p=0.9`：只从累积概率90%的词中选，避免低概率噪声  
- `sample_top_p` 函数用随机数+累计和实现采样

你终于明白：**AI的创造力，来自于概率的随机性**。

---

### 结语：1000行，胜过千篇论文

llama.c不是用来部署的，它是用来**理解**的。

当你在Colab里跑一个`pipeline("text-generation")`时，你只是在调用一个黑箱。  
而当你读完llama.c，你**亲手推导了Transformer的每一层**，你**看到了RoPE如何编码位置**，你**明白了采样如何产生“灵感”**。

这不是“学习AI”，这是**成为AI的造物主**。

> “你不必成为专家才能理解它。  
> 你只需要，从最简单的代码开始。”  
> —— Andrej Karpathy

---

📌 **项目地址**：https://github.com/karpathy/llama.c  
📌 **建议你**：  
1. 下载代码  
2. `gcc -O2 llama.c -o llama -lm`  
3. 下载7B模型权重  
4. 运行 `./llama "The meaning of life is "`  
5. 看着AI一个字一个字地“想”出来

你会发现：  
**AI，从来不是魔法。**  
**它只是，数学的舞蹈。**

