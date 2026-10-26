# Qwen Programming Tutor

A guided, hands-on project for learning how to instruction-tune a conversational language model with PyTorch, Hugging Face, and LoRA.

This repository is intentionally a work in progress. Each lesson will be implemented incrementally so the code is understood rather than generated all at once.

## Goal

Fine-tune a small Qwen instruction model into a beginner-friendly programming tutor, evaluate it against the original model, and build a local chat interface.

## Planned lessons

1. Define the task and run the base model
2. Explore a public instruction dataset
3. Format conversations with the Qwen chat template
4. Mask prompt tokens so training targets only assistant responses
5. Configure LoRA and GPU-efficient training
6. Fine-tune and save an adapter
7. Evaluate the base and tuned models on unseen prompts
8. Build a local interactive chat interface

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

Repository setup is complete. Lesson 1 is the next step.
