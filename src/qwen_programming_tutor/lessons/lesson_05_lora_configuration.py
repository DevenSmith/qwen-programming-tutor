import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


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

lora_configuration = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

print("\nLoRA configuration:")
print(lora_configuration)

model = get_peft_model(
    model,
    lora_configuration,
)

total_parameters_after_lora = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters_after_lora = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

trainable_percentage = (
    100
    * trainable_parameters_after_lora
    / total_parameters_after_lora
)

print("\nTotal parameters after LoRA:")
print(f"{total_parameters_after_lora:,}")

print("\nTrainable parameters after LoRA:")
print(f"{trainable_parameters_after_lora:,}")

print("\nTrainable percentage:")
print(f"{trainable_percentage:.4f}%")

print("\nFirst trainable LoRA parameters:")

shown_parameters = 0

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(f"{name}: {tuple(parameter.shape)}")
        shown_parameters += 1

        if shown_parameters == 8:
            break