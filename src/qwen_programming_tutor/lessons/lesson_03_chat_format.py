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