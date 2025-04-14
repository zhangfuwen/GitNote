
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

