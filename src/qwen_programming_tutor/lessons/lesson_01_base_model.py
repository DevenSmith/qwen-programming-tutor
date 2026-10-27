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

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
).to(device)

model.eval()

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nNumber of parameters:")
print(f"{number_of_parameters:,}")

print("\nModel device:")
print(next(model.parameters()).device)

print("\nModel data type:")
print(next(model.parameters()).dtype)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False,
    )

response_ids = generated_ids[
    0,
    input_ids.shape[1] :,
]

response = tokenizer.decode(
    response_ids,
    skip_special_tokens=True,
)

print("\nBase model response:")
print(response)