import torch
from transformers import AutoModelForCausalLM


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
).to(device)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nDevice:")
print(device)

print("\nTotal parameters:")
print(f"{total_parameters:,}")

print("\nTrainable parameters before LoRA:")
print(f"{trainable_parameters:,}")