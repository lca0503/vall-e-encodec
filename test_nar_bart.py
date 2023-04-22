from transformers import AutoTokenizer
from nar_bart import NARBartForConditionalGeneration
import torch

model = NARBartForConditionalGeneration.from_pretrained("voidful/bart-base-unit")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")


ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
# Generate Summary
output = model(inputs["input_ids"])
outputs = torch.max(output.logits, dim=-1).indices
x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(x)
