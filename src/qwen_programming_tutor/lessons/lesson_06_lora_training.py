from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
maximum_sequence_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model:")
print(model_name)

print("\nMaximum sequence length:")
print(maximum_sequence_length)

print("\nPadding token:")
print(repr(tokenizer.pad_token))

print("\nPadding token ID:")
print(tokenizer.pad_token_id)

print("\nEnd-of-sequence token:")
print(repr(tokenizer.eos_token))

print("\nEnd-of-sequence token ID:")
print(tokenizer.eos_token_id)

number_of_training_examples = 200
number_of_validation_examples = 40
number_of_selected_examples = (
    number_of_training_examples
    + number_of_validation_examples
)


def passed_all_tests(example):
    return float(example["average_test_score"]) == 1.0

def prepare_example(example):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a patient programming tutor. "
                "Provide correct Python solutions with clear explanations."
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

    full_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )

    full_input_ids = full_inputs["input_ids"][0]

    prompt_inputs = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    prompt_input_ids = prompt_inputs["input_ids"][0]

    input_ids = full_input_ids[:maximum_sequence_length]
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    prompt_length = min(
        prompt_input_ids.shape[0],
        maximum_sequence_length,
    )

    labels[:prompt_length] = -100

    if torch.all(labels == -100):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("\nLoading dataset...")

dataset = load_dataset(
    "nvidia/OpenCodeInstruct",
    split="train",
    streaming=True,
)

filtered_dataset = dataset.filter(passed_all_tests)

shuffled_dataset = filtered_dataset.shuffle(
    seed=42,
    buffer_size=1_000,
)

selected_examples = list(
    shuffled_dataset.take(number_of_selected_examples)
)

training_examples = selected_examples[:number_of_training_examples]
validation_examples = selected_examples[number_of_training_examples:]

print("\nSelected examples:")
print(len(selected_examples))

print("\nTraining examples:")
print(len(training_examples))

print("\nValidation examples:")
print(len(validation_examples))

prepared_example = None

for example in training_examples:
    prepared_example = prepare_example(example)

    if prepared_example is not None:
        break

print("\nPrepared input tokens:")
print(prepared_example["input_ids"].shape[0])

print("\nIgnored prompt tokens:")
print((prepared_example["labels"] == -100).sum().item())

print("\nAssistant tokens used for training:")
print((prepared_example["labels"] != -100).sum().item())

def prepare_examples(examples):
    prepared_examples = []

    for example in examples:
        prepared = prepare_example(example)

        if prepared is not None:
            prepared_examples.append(prepared)

    return prepared_examples

prepared_training_examples = prepare_examples(
    training_examples
)

prepared_validation_examples = prepare_examples(
    validation_examples
)

print("\nUsable training examples:")
print(len(prepared_training_examples))

print("\nDiscarded training examples:")
print(
    len(training_examples)
    - len(prepared_training_examples)
)

print("\nUsable validation examples:")
print(len(prepared_validation_examples))

print("\nDiscarded validation examples:")
print(
    len(validation_examples)
    - len(prepared_validation_examples)
)


def collate_batch(examples):
    input_ids = pad_sequence(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_mask = pad_sequence(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=0,
    )

    labels = pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

batch_size = 1

training_loader = DataLoader(
    prepared_training_examples,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)

validation_loader = DataLoader(
    prepared_validation_examples,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)

first_batch = next(iter(training_loader))

print("\nTraining batches:")
print(len(training_loader))

print("\nFirst input batch shape:")
print(first_batch["input_ids"].shape)

print("\nFirst attention-mask shape:")
print(first_batch["attention_mask"].shape)

print("\nFirst label batch shape:")
print(first_batch["labels"].shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
).to(device)

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

model = get_peft_model(
    model,
    lora_configuration,
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nDevice:")
print(device)

print("\nTrainable LoRA parameters:")
print(f"{trainable_parameter_count:,}")

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=2e-4,
)

first_batch = {
    name: tensor.to(device)
    for name, tensor in first_batch.items()
}

model.eval()

with torch.no_grad():
    initial_outputs = model(**first_batch)
    initial_loss = initial_outputs.loss.item()

model.train()

print("\nInitial batch loss:")
print(initial_loss)

print("\nOptimizer parameters:")
print(
    f"{sum(parameter.numel() for parameter in trainable_parameters):,}"
)

@torch.no_grad()
def calculate_validation_loss(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_trained_tokens = 0

    for batch in data_loader:
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }

        outputs = model(**batch)

        trained_token_count = (
            batch["labels"][:, 1:] != -100
        ).sum().item()

        total_loss += (
            outputs.loss.item()
            * trained_token_count
        )

        total_trained_tokens += trained_token_count

    model.train()

    return total_loss / total_trained_tokens

initial_validation_loss = calculate_validation_loss(
    model=model,
    data_loader=validation_loader,
    device=device,
)

print("\nInitial validation loss:")
print(initial_validation_loss)

number_of_epochs = 1
gradient_accumulation_steps = 8
maximum_gradient_norm = 1.0
logging_interval = 25

optimizer.zero_grad()

for epoch in range(number_of_epochs):
    model.train()

    accumulated_loss = 0.0
    accumulated_tokens = 0

    for batch_index, batch in enumerate(training_loader):
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        trained_token_count = (
            batch["labels"][:, 1:] != -100
        ).sum().item()

        accumulated_loss += (
            loss.item()
            * trained_token_count
        )
        accumulated_tokens += trained_token_count

        scaled_loss = (
            loss
            / gradient_accumulation_steps
        )
        scaled_loss.backward()

        finished_accumulation = (
            (batch_index + 1)
            % gradient_accumulation_steps
            == 0
        )

        finished_epoch = (
            batch_index + 1
            == len(training_loader)
        )

        if finished_accumulation or finished_epoch:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=maximum_gradient_norm,
            )

            optimizer.step()
            optimizer.zero_grad()

        if (
            (batch_index + 1) % logging_interval == 0
            or finished_epoch
        ):
            average_loss = (
                accumulated_loss
                / accumulated_tokens
            )

            print(
                f"epoch={epoch + 1} "
                f"batch={batch_index + 1:3d}/"
                f"{len(training_loader)} "
                f"training_loss={average_loss:.4f}"
            )

final_validation_loss = calculate_validation_loss(
    model=model,
    data_loader=validation_loader,
    device=device,
)

validation_improvement = (
    initial_validation_loss
    - final_validation_loss
)

print("\nInitial validation loss:")
print(initial_validation_loss)

print("\nFinal validation loss:")
print(final_validation_loss)

print("\nValidation loss improvement:")
print(validation_improvement)

project_directory = Path(__file__).resolve().parents[3]

adapter_directory = (
    project_directory
    / "checkpoints"
    / "qwen_programming_tutor_lora"
)

model.save_pretrained(adapter_directory)
tokenizer.save_pretrained(adapter_directory)

saved_file_size = sum(
    path.stat().st_size
    for path in adapter_directory.rglob("*")
    if path.is_file()
)

print("\nSaved LoRA adapter:")
print(adapter_directory)

print("\nSaved files:")
for path in sorted(adapter_directory.iterdir()):
    if path.is_file():
        print(path.name)

print("\nTotal saved size:")
print(f"{saved_file_size / 1_000_000:.2f} MB")