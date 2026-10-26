import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:")
print(device)

print("\nModel:")
print(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\nVocabulary size:")
print(len(tokenizer))

messages = [
    {
        "role": "system",
        "content": "You are a patient programming tutor. Explain concepts clearly and concisely.",
    },
    {
        "role": "user",
        "content": "What does optimizer.step() do in PyTorch?",
    },
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nFormatted prompt:")
print(formatted_prompt)

model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(device)

input_ids = model_inputs["input_ids"]

print("\nInput shape:")
print(input_ids.shape)

print("\nNumber of input tokens:")
print(input_ids.shape[1])

print("\nInput device:")
print(input_ids.device)

print("\nFirst 20 token IDs:")
print(input_ids[0, :20])