
# 模型结构

## 基本结构

## Attention



MHA

MQA

GQA

MLA

![](assets/Pasted%20image%2020250313144307.png)

## MoE


## LoRA



# 模型训练

## Llama

### The Llama 3 Herd of Models

![](assets/468347782_9231729823505907_4580471254289036098_n.pdf)

#### pretrainin



#### post-training

RL

SFT

DPO

## Apple Intelligence Foundation Models

https://machinelearning.apple.com/research/introducing-apple-foundation-models

![](assets/AppleIntelligenceFoundationModels_2407.21075v1.pdf)

![](assets/Pasted%20image%2020250314101343.png)
AFM-on-device模型参数量2.58B（0.15B embedding)，推理速度是`0.6 ms per prompt token`，30 token/s without token speculation。

优化点：

1. Shared input/output embedding
2. GQA: 24 query, 8 kv heads
3. LoRA adpater on-the-fly，rank 16的adapter大小在10MB量级。
4. 量化：4比特、2比特混合量化，总体小于4bits， 3.7bpw。GPTQ、AWQ。
5. Accuracy recovery adapter

Device model用于：
1. 便签中写作、校对、总结等场景。
2. 邮件、短信、通知的优先级+紧急程度判断
3. 邮件总结、回复、语气调整

![](assets/Pasted%20image%2020250314104200.png)




## DeepSeek


### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

https://arxiv.org/abs/2501.12948


![](assets/2501.12948v1.pdf)

![](assets/Pasted%20image%2020250313142917.png)

https://arxiv.org/abs/2402.03300

![](assets/Pasted%20image%2020250313142934.png)

![](assets/Pasted%20image%2020250313143017.png)

![](assets/Pasted%20image%2020250313143032.png)

![](assets/Pasted%20image%2020250313143048.png)

![](assets/Pasted%20image%2020250313143107.png)

# 推理


# Agent

