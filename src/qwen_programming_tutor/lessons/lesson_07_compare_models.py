from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

project_directory = Path(__file__).resolve().parents[3]

adapter_directory = (
    project_directory
    / "checkpoints"
    / "qwen_programming_tutor_lora"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if not adapter_directory.exists():
    raise FileNotFoundError(
        f"LoRA adapter was not found: {adapter_directory}"
    )

print("Device:")
print(device)

print("\nBase model:")
print(model_name)

print("\nLoRA adapter:")
print(adapter_directory)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_directory
)

system_instruction = (
    "You are a patient programming tutor. "
    "Provide correct Python solutions with clear explanations."
)

evaluation_prompts = [
    """
A beginner wrote this Python function:

def add_item(item, items=[]):
    items.append(item)
    return items

Explain the bug, show corrected code, and keep the answer under 180 words.
""".strip(),
    """
Write a Python function that returns the first character that occurs only
once in a string. Explain the time complexity and include two examples.
Keep the answer under 180 words.
""".strip(),
    """
Explain the difference between a shallow copy and a deep copy in Python.
Include a short nested-list example demonstrating why the distinction
matters. Keep the answer under 180 words.
""".strip(),
]


def create_messages(user_prompt):
    return [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


messages = create_messages(
    evaluation_prompts[0]
)

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nFormatted evaluation prompt:")
print(formatted_prompt)


def generate_response(model, messages):
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    prompt_length = model_inputs["input_ids"].shape[1]

    model.eval()

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = generated_ids[
        0,
        prompt_length:,
    ]

    return tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )


def generate_responses(model):
    responses = []

    for user_prompt in evaluation_prompts:
        prompt_messages = create_messages(user_prompt)

        response = generate_response(
            model=model,
            messages=prompt_messages,
        )

        responses.append(response)

    return responses


print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
).to(device)

base_responses = generate_responses(
    model=base_model,
)

print("\nLoading LoRA adapter...")

tuned_model = PeftModel.from_pretrained(
    base_model,
    adapter_directory,
)

tuned_responses = generate_responses(
    model=tuned_model,
)


def print_response_summary(name, response):
    word_count = len(response.split())

    includes_correct_fix = (
        "items=None" in response
        or "items = None" in response
    )

    explains_shared_default = (
        "same list" in response.lower()
        or "reused" in response.lower()
        or "created once" in response.lower()
    )

    print(f"\n{name} summary:")
    print(f"Word count: {word_count}")
    print(f"Within 180 words: {word_count <= 180}")
    print(f"Includes corrected code: {includes_correct_fix}")
    print(
        "Explains reused default list: "
        f"{explains_shared_default}"
    )


for index, user_prompt in enumerate(
    evaluation_prompts
):
    print("\n" + "=" * 70)
    print(f"EVALUATION PROMPT {index + 1}")
    print("=" * 70)

    print("\nUser prompt:")
    print(user_prompt)

    print("\nBase-model response:")
    print(base_responses[index])

    print("\nFine-tuned response:")
    print(tuned_responses[index])

    if index == 0:
        print_response_summary(
            "Base model",
            base_responses[index],
        )

        print_response_summary(
            "Fine-tuned model",
            tuned_responses[index],
        )
