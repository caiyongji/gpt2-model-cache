# gpt2-model-cache

gpt2 模型下载，源自transformers



```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')
generated = tokenizer.encode("REPLACE HERE................................................")
context = torch.tensor([generated])
past_key_values = None
for i in range(30):
    output = model(context, past_key_values=past_key_values)
    past_key_values = output.past_key_values
    token = torch.argmax(output.logits[..., -1, :])
    context = token.unsqueeze(0)
    generated += [token.tolist()]
sequence = tokenizer.decode(generated)
sequence = sequence.split(".")[:-1]
print(sequence)


```
