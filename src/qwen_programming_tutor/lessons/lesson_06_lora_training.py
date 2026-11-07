from transformers import AutoTokenizer


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