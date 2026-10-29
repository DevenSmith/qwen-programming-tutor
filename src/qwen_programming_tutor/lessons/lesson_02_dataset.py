from datasets import load_dataset


dataset_name = "nvidia/OpenCodeInstruct"

dataset = load_dataset(
    dataset_name,
    split="train",
    streaming=True,
)

print("Dataset:")
print(dataset)

print("\nColumns:")
print(dataset.column_names)

example = next(iter(dataset))

print("\nExample ID:")
print(example["id"])

print("\nDomain:")
print(example["domain"])

print("\nGeneration algorithm:")
print(example["generation_algorithm"])

print("\nAverage test score:")
print(example["average_test_score"])

print("\nInput character count:")
print(len(example["input"]))

print("\nOutput character count:")
print(len(example["output"]))

print("\nInput:")
print(example["input"][:1000])

print("\nOutput:")
print(example["output"][:1000])