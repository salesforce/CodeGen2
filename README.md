# CodeGen2

Official research release for the **CodeGen2** models (`3B`, `7B`, `16B`) for **Program Synthesis** as presented in ICLR 2023:

*Title*: [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/TBD)

*Authors*: [Erik Nijkamp](https://enijkamp.github.io/)\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ&hl=en), [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en), and [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en) (* indicates equal contribution)

## Sampling

Program synthesis in the form of auto-regressive sampling can be performed as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained()
model = AutoModelForCausalLM.from_pretrained()
inputs = tokenizer("# this function prints hello world", return_tensors="pt").to(0)
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
```
