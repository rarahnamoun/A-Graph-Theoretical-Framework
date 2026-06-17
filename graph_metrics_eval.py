import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_STEPS = [
    "step3000",
    "step18000",
    "step33000",
    "step47000",
    "step62000",
    "step77000",
    "step92000",
    "step106000",
    "step121000",
    "step143000"
]

ALL_METRICS = ["spectral_entropy", "graph_energy", "density"]


def spectral_entropy(G):
    A = nx.adjacency_matrix(G).todense()
    degrees = np.sum(A, axis=1).flatten()
    L = np.diag(degrees) - A
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    total = np.sum(eigenvalues)
    if total == 0:
        return 0.0
    probs = eigenvalues / total
    return float(-np.sum(probs * np.log(probs)))


def graph_energy(G):
    A = nx.adjacency_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(A)
    return float(np.sum(np.abs(eigenvalues)))


def graph_density(G):
    return nx.density(G)


METRIC_FUNCTIONS = {
    "spectral_entropy": spectral_entropy,
    "graph_energy": graph_energy,
    "density": graph_density
}


def get_metrics_from_graph(G, metrics):
    if len(G) == 0:
        return {m: 0.0 for m in metrics}
    return {m: METRIC_FUNCTIONS[m](G) for m in metrics}


def build_word_transition_graph(prompt, model, tokenizer, token_limit, top_k):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + token_limit,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        output_scores=True,
        return_dict_in_generate=True
    )
    generated_ids = outputs.sequences[0]
    output_tokens = generated_ids[len(inputs.input_ids[0]):]
    generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    G = nx.DiGraph()
    scores = outputs.scores
    input_length = inputs.input_ids.shape[1]
    for i, logits in enumerate(scores):
        probs = F.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)
        prev_token_id = inputs.input_ids[0, -1] if i == 0 else generated_ids[input_length + i - 1]
        prev_word = tokenizer.decode(prev_token_id).strip()
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            candidate_word = tokenizer.decode(token_id).strip()
            G.add_node(prev_word)
            G.add_node(candidate_word)
            G.add_edge(prev_word, candidate_word, weight=prob)
    return G, generated_text


def load_model(model_path, revision=None, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, cache_dir=cache_dir, trust_remote_code=True
    ).eval().to(device)
    return tokenizer, model


def load_raw_dataset(dataset_name):
    if dataset_name == "logiqa":
        dataset = load_dataset("lucasmccabe/logiqa", split="validation")
        validation = [
            {
                "context": sample["context"].strip(),
                "query": sample["query"].strip(),
                "options": sample["options"],
                "correct_answer": sample["options"][sample["correct_option"]].strip()
            }
            for sample in dataset
        ]
    elif dataset_name == "piqa":
        dataset = load_dataset("ybisk/piqa", cache_dir=".")
        validation = [
            {
                "goal": sample["goal"].strip(),
                "options": [sample["sol1"].strip(), sample["sol2"].strip()],
                "correct_answer": str(sample["label"])
            }
            for sample in dataset["validation"]
        ]
    else:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=".")
        validation = [
            {
                "question": sample["question"].strip(),
                "options": [opt.strip() for opt in sample["choices"]["text"]],
                "correct_answer": str(sample["answerKey"])
            }
            for sample in dataset["test"]
        ]
    return validation


def format_predefined_entry(dataset_name, entry):
    if dataset_name == "logiqa":
        prompt = f"CONTEXT:\n{entry['context']}\n\nQUESTION:\n{entry['query']}\n\nOPTIONS:\n"
        for idx, opt in enumerate(entry["options"]):
            prompt += f"{idx}: {opt}\n"
        references = [entry["correct_answer"]]
    elif dataset_name == "piqa":
        prompt = f"GOAL:\n{entry['goal']}\n\nSolution 1:\n{entry['options'][0]}\nSolution 2:\n{entry['options'][1]}"
        references = [entry["correct_answer"]]
    else:
        option_lines = "\n".join(f"Option {i + 1}:\n{opt}" for i, opt in enumerate(entry["options"]))
        prompt = f"Question:\n{entry['question']}\n\n{option_lines}"
        references = [entry["correct_answer"]]
    return prompt, references


def load_predefined_samples(dataset_name, num_samples, seed):
    pool = load_raw_dataset(dataset_name)
    random.seed(seed)
    num_samples = min(num_samples, len(pool))
    selected = random.sample(pool, num_samples)
    samples = []
    for entry in selected:
        prompt, references = format_predefined_entry(dataset_name, entry)
        samples.append({"prompt": prompt, "references": references})
    return samples


def load_custom_samples(path):
    with open(path, "r") as f:
        data = json.load(f)
    samples = data["samples"]
    for sample in samples:
        if "prompt" not in sample:
            raise ValueError("Every entry in the custom dataset must include a 'prompt' field")
        sample.setdefault("references", [])
    return samples


def plot_metrics_over_steps(steps, metrics_by_step, output_dir):
    for metric_name, values in metrics_by_step.items():
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker="o")
        plt.title(f"{metric_name.replace('_', ' ').title()} Across Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_over_steps.png"))
        plt.close()


def plot_metrics_bar_chart(all_metrics, output_dir):
    if not all_metrics:
        return
    metric_names = list(next(iter(all_metrics.values())).keys())
    for metric in metric_names:
        plt.figure(figsize=(12, 6))
        values = [all_metrics[model][metric] for model in all_metrics]
        plt.bar(list(all_metrics.keys()), values)
        plt.title(f"{metric.replace('_', ' ').title()} per Model")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_bar_chart.png"))
        plt.close()


def run_checkpoints_mode(args, samples):
    os.makedirs(args.output_dir, exist_ok=True)
    model_base = f"{args.model_base}-{args.model_size}-deduped"
    cache_base_path = os.path.join(args.cache_dir, model_base.replace("/", "_"))

    step_numbers = []
    metrics_by_step = {m: [] for m in args.metrics}

    for step in args.steps:
        print(f"\n=== Running checkpoint: {model_base} @ {step} ===")
        step_numbers.append(int(step.replace("step", "")))

        tokenizer, model = load_model(
            model_base, revision=step, cache_dir=os.path.join(cache_base_path, step)
        )

        step_metrics = {m: [] for m in args.metrics}
        outputs = {}
        for sample in samples:
            G, generated_text = build_word_transition_graph(
                sample["prompt"], model, tokenizer, args.token_limit, args.top_k
            )
            metrics = get_metrics_from_graph(G, args.metrics)
            outputs[sample["prompt"]] = {
                "generated": generated_text,
                "references": sample.get("references", [])
            }
            for m, v in metrics.items():
                step_metrics[m].append(v)
            for m, v in metrics.items():
                print(f"{m}: {v}")

        avg_metrics = {m: float(np.mean(v)) for m, v in step_metrics.items()}
        with open(os.path.join(args.output_dir, f"metrics_{step}.json"), "w") as f:
            json.dump(avg_metrics, f, indent=2)
        with open(os.path.join(args.output_dir, f"outputs_{step}.json"), "w") as f:
            json.dump(outputs, f, indent=2)

        for m in args.metrics:
            metrics_by_step[m].append(avg_metrics[m])

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "metrics_all_steps.json"), "w") as f:
        json.dump({"steps": step_numbers, "metrics": metrics_by_step}, f, indent=2)

    plot_metrics_over_steps(step_numbers, metrics_by_step, args.output_dir)


def run_models_mode(args, samples):
    os.makedirs(args.output_dir, exist_ok=True)
    all_model_metrics = {}

    for model_path in args.models:
        print(f"\n=== Running model: {model_path} ===")
        try:
            tokenizer, model = load_model(model_path, cache_dir=args.cache_dir)

            sample_metrics = {m: [] for m in args.metrics}
            outputs = {}
            for sample in samples:
                G, generated_text = build_word_transition_graph(
                    sample["prompt"], model, tokenizer, args.token_limit, args.top_k
                )
                metrics = get_metrics_from_graph(G, args.metrics)
                outputs[sample["prompt"]] = {
                    "generated": generated_text,
                    "references": sample.get("references", [])
                }
                for m, v in metrics.items():
                    sample_metrics[m].append(v)

            avg_metrics = {m: float(np.mean(v)) for m, v in sample_metrics.items()}
            all_model_metrics[model_path] = avg_metrics

            sanitized = model_path.replace("/", "_")
            with open(os.path.join(args.output_dir, f"metrics_{sanitized}.json"), "w") as f:
                json.dump(avg_metrics, f, indent=2)
            with open(os.path.join(args.output_dir, f"outputs_{sanitized}.json"), "w") as f:
                json.dump(outputs, f, indent=2)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing model {model_path}: {e}")

    with open(os.path.join(args.output_dir, "metrics_all_models.json"), "w") as f:
        json.dump(all_model_metrics, f, indent=2)

    plot_metrics_bar_chart(all_model_metrics, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generation-time graph metrics across model checkpoints or across different models."
    )
    parser.add_argument("--mode", choices=["checkpoints", "models"], required=True)
    parser.add_argument(
        "--dataset",
        choices=["logiqa", "piqa", "arc", "custom"],
        required=True
    )
    parser.add_argument("--custom_dataset", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_base", type=str, default="EleutherAI/pythia")
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--steps", nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--token_limit", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=ALL_METRICS)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cache_dir", type=str, default=".")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "custom":
        if not args.custom_dataset:
            raise SystemExit("--custom_dataset is required when --dataset custom is selected")
        samples = load_custom_samples(args.custom_dataset)
    else:
        samples = load_predefined_samples(args.dataset, args.num_samples, args.seed)

    if args.mode == "checkpoints":
        if not args.model_size:
            raise SystemExit("--model_size is required when --mode checkpoints is selected")
        run_checkpoints_mode(args, samples)
    else:
        if not args.models:
            raise SystemExit("--models is required when --mode models is selected")
        run_models_mode(args, samples)


if __name__ == "__main__":
    main()
