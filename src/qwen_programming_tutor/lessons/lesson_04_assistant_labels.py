import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_name = "nvidia/OpenCodeInstruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

dataset = load_dataset(
    dataset_name,
    split="train",
    streaming=True,
)


def passed_all_tests(example):
    return float(example["average_test_score"]) == 1.0


filtered_dataset = dataset.filter(
    passed_all_tests
)

example = next(
    iter(filtered_dataset)
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a patient programming tutor. "
            "Provide correct Python solutions with clear docstrings."
        ),
    },
    {
        "role": "user",
        "content": example["input"],
    },
    {
        "role": "assistant",
        "content": example["output"],
    },
]

for message in messages:
    print("\nRole:")
    print(message["role"])

    print("Content preview:")
    print(message["content"][:300])

    formatted_conversation = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print("\nFormatted conversation:")
print(formatted_conversation)

model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    return_dict=True,
)

input_ids = model_inputs["input_ids"]
labels = input_ids.clone()

print("Input shape:")
print(input_ids.shape)

print("\nLabel shape:")
print(labels.shape)

print("\nInputs and labels initially identical:")
print(torch.equal(input_ids, labels))