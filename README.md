# Qwen Programming Tutor

A guided, hands-on project for learning how to instruction-tune a conversational language model with PyTorch, Hugging Face, and LoRA.

This repository is intentionally a work in progress. Each lesson will be implemented incrementally so the code is understood rather than generated all at once.

## Goal

Fine-tune a small Qwen instruction model into a beginner-friendly programming tutor, evaluate it against the original model, and build a local chat interface.

## Course roadmap

1. Define the task and run the base model
2. Explore a public instruction dataset
3. Format conversations with the Qwen chat template
4. Mask prompt tokens so training targets only assistant responses
5. Configure LoRA and GPU-efficient training
6. Fine-tune and save an adapter
7. Evaluate the base and tuned models on unseen prompts
8. Build a local interactive chat interface

## Completed lessons

### Lesson 1: Run and evaluate the base model

Loaded `Qwen/Qwen2.5-1.5B-Instruct` with Hugging Face Transformers, placed it on the CUDA GPU in half precision, formatted a programming question with Qwen's chat template, and generated a baseline response.

This lesson also introduced:

- The difference between tokens and words
- Why `generate()` returns both the prompt and newly generated tokens
- How to slice out and decode only the model's response
- Why inference should not build a gradient graph
- Simple automatic checks for response length and natural completion

The baseline evaluation showed why fine-tuning is useful: the model's response was verbose, ended mid-answer, and included inaccurate statements about `optimizer.step()`.

### Lesson 2: Explore and filter an instruction dataset

Loaded NVIDIA's `OpenCodeInstruct` dataset in streaming mode so examples could be inspected without downloading the entire dataset first. Each example contains a programming request, a proposed answer, and quality metadata such as test results and LLM judgements.

This lesson demonstrated that a high score does not guarantee a correct answer. One example passed 9 of 10 tests because its greedy scheduling solution incorrectly assumed that tasks sorted by start time could always be selected in that order. The dataset was then filtered to examples with an `average_test_score` of `1.0`.

Main takeaway: dataset quality must be investigated rather than assumed. Automated tests and model-based judgements are useful signals, but neither is an absolute guarantee of correctness or teaching quality.

### Lesson 3: Format training conversations

Converted a filtered dataset example into the three roles expected by an instruction model:

- `system`: defines the model's behavior as a patient programming tutor
- `user`: contains the programming problem
- `assistant`: contains the desired answer

Qwen's chat template converted those messages into its required structure using special markers such as `<|im_start|>` and `<|im_end|>`. The formatted conversation was then tokenized into `input_ids` and an `attention_mask`.

Main takeaway: a model does not receive Python dictionaries or role names directly. The chat template turns structured messages into the exact token sequence the model was designed to understand.

### Lesson 4: Train only on assistant responses

Created labels by cloning the complete conversation's `input_ids`, then measured the token length of the system-and-user prompt. The first 216 labels were replaced with `-100`, leaving 135 assistant tokens as training targets.

PyTorch's causal language-model loss ignores positions labeled `-100`. The model can therefore read the complete prompt as context while being graded only on its prediction of the assistant's response. The final `<|im_end|>` token remains a target so the model also learns when to stop answering.

Main takeaway: the input and training target serve different purposes. The model reads all conversation tokens, but only the assistant portion contributes to the loss. Hugging Face performs the one-token causal prediction shift internally, so the labels should not be shifted manually.

## Planned technologies

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- PEFT and LoRA
- Qwen2.5-1.5B-Instruct
- CUDA

## Project structure

```text
qwen-programming-tutor/
├── src/qwen_programming_tutor/
│   └── lessons/                 # Added one lesson at a time
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Status

Lessons 1 through 4 are complete. The project can load and evaluate the base Qwen model, inspect and filter instruction data, apply Qwen's chat format, and construct assistant-only training labels.

The next step is Lesson 5: configure LoRA and GPU-efficient training.
