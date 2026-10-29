import ast

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

unit_tests = ast.literal_eval(
    example["unit_tests"]
)

test_statuses = ast.literal_eval(
    example["tests_execution_status"]
)

print("\nNumber of tests:")
print(len(unit_tests))

print("\nPassed tests:")
print(test_statuses.count("pass"))

print("\nFailed tests:")
print(test_statuses.count("fail"))

print("\nTests and results:")

for test, status in zip(
    unit_tests,
    test_statuses,
    strict=True,
):
    print(status.upper())
    print(test.strip())

def passed_all_tests(example):
    return float(example["average_test_score"]) == 1.0


filtered_dataset = dataset.filter(
    passed_all_tests
)

filtered_example = next(
    iter(filtered_dataset)
)

print("\nFirst filtered example:")

print("ID:")
print(filtered_example["id"])

print("Test score:")
print(filtered_example["average_test_score"])

print("Input:")
print(filtered_example["input"][:500])

print("Output:")
print(filtered_example["output"][:500])    