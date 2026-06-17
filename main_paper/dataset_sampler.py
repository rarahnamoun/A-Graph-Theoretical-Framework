import json
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import random

def tokenize(text):
    return set(text.lower().split())


def load_data(dataset_name):
    if dataset_name == "trivia_qa":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", cache_dir='.')
        data = [
            {
                "question": sample["question"].strip(),
                "answers": [answer.strip() for answer in sample["answer"]["normalized_aliases"]],
                "preferred_answer": sample["answer"]["value"].strip()
            }
            for sample in dataset["train"]
            if len(sample["answer"]["normalized_aliases"]) > 0
        ]
    elif dataset_name == "logiqa":
        dataset = load_dataset("lucasmccabe/logiqa", split="validation")
        data = [
            {
                "context": sample["context"].strip(),
                "query": sample["query"].strip(),
                "options": sample["options"],
                "correct_answer": sample["options"][sample["correct_option"]].strip()
            }
            for sample in dataset
        ]
    elif dataset_name == "piqa":
        dataset = load_dataset("ybisk/piqa", cache_dir='.')
        data = [
            {
                "goal": sample["goal"].strip(),
                "options": [sample["sol1"].strip(), sample["sol2"].strip()],
                "correct_answer": sample["sol1"].strip() if sample["label"] == 0 else sample["sol2"].strip()
            }
            for sample in dataset["validation"]
        ]
    elif dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir='.')
        data = [
            {
                "question": sample["question"].strip(),
                "options": [opt.strip() for opt in sample["choices"]["text"]],
                "correct_answer": sample["choices"]["text"][ord(sample["answerKey"]) - ord("A")].strip()
            }
            for sample in dataset["test"]
            if sample["answerKey"] in ["A", "B", "C", "D"]
        ]
    else:  # nq_open
        dataset = load_dataset("nq_open", cache_dir='.')
        data = [
            {
                "question": sample["question"].strip(),
                "answers": [answer.strip() for answer in sample["answer"]],
                "preferred_answer": sample["answer"][0].strip()
            }
            for sample in dataset["train"]
        ]

    return data

def evaluate_and_save(num_samples=3):
    dataset_names = [ "logiqa", "piqa", "arc"]

    for name in dataset_names:
        print(f"\n=== Dataset: {name.upper()} ===")
        data = load_data(name)
        sampled_data = random.sample(data, num_samples)  # Randomly sample data
        samples = []

        for sample in sampled_data:
            prompt_parts = []
            if name == "logiqa":
                prompt_parts.append(sample["context"])
                prompt_parts.append(sample["query"])
                for idx, opt in enumerate(sample["options"], 1):
                    prompt_parts.append(f"Choice {idx}: {opt}")
                prompt = " ".join(prompt_parts)
                references = [sample["correct_answer"]]
            elif name == "piqa":
                prompt_parts.append(sample["goal"])
                for idx, opt in enumerate(sample["options"], 1):
                    prompt_parts.append(f"Choice {idx}: {opt}")
                prompt = " ".join(prompt_parts)
                references = [sample["correct_answer"]]
            elif name == "arc":
                prompt_parts.append(sample["question"])
                for idx, opt in enumerate(sample["options"], 1):
                    prompt_parts.append(f"Choice {idx}: {opt}")
                prompt = " ".join(prompt_parts)
                references = [sample["correct_answer"]]
            else:
                continue

            samples.append({
                "prompt": prompt,
                "references": references
            })

        output_path = f"{name}_samples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": name,
                "samples": samples
            }, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(samples)} samples to: {output_path}")

# Ask user for number of samples (or default to 3)
if __name__ == "__main__":
    try:
        num = int(input("How many samples per dataset? (default 3): ") or 3)
    except ValueError:
        num = 3
    evaluate_and_save(num_samples=num)