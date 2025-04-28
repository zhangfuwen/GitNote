
## 推理引擎关键问题

1. 内存容量
	1. 权重
		1. 量化
		2. 蒸馏
	2. kv cache
		1. 缓存 sglang的radix tree
		2. 内存碎片化 paged attention
2. 算力
	1. prefill阶段是计算密集型的，核心是如何充分利用硬件资源
3. 内存带宽
	1. decode阶段是访存密集型的，核心是如何加载一次权重，计算更多token
	2. batching
	3. 预测性推理 [万字综述 10+ 种 LLM 投机采样推理加速方案 - 53AI-AI知识库\|大模型知识库\|大模型训练\|智能体开发](https://www.53ai.com/news/finetuning/2024071109285.html)
		1. 投机采样
		2. 多头美杜莎
		3. LookAHead  [GitHub - hao-ai-lab/LookaheadDecoding: \[ICML 2024\] Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding)

## 效果

### 思考更长时间

Think Twice

[Think Longer](https://www.alphaxiv.org/abs/2503.23803)
