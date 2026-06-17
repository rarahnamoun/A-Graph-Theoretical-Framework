import torch
import json
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from typing import List

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

def build_word_transition_graph(prompt, model, tokenizer, device, token_limit=40):
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
        top3_probs, top3_indices = torch.topk(probs, 60)

        selected_token_id = generated_ids[input_length + i]
        prev_token_id = inputs.input_ids[0, -1] if i == 0 else generated_ids[input_length + i - 1]
        prev_word = tokenizer.decode(prev_token_id).strip()

        for prob, token_id in zip(top3_probs.tolist(), top3_indices.tolist()):
            candidate_word = tokenizer.decode(token_id).strip()
            selected_flag = (token_id == selected_token_id.item())
            G.add_node(prev_word)
            G.add_node(candidate_word)
            G.add_edge(prev_word, candidate_word, weight=prob, selected=selected_flag)

    return G, generated_text

def get_spectral_metrics_from_graph(G):
    if len(G) == 0:
        return {m: 0.0 for m in [
            "spectral_entropy", "graph_energy", "density"
        ]}

    return {
        "spectral_entropy": spectral_entropy(G),
        "graph_energy": graph_energy(G),
        "density": graph_density(G)
    }

def load_data(dataset_name):
   if dataset_name == "logiqa":
        dataset = load_dataset("lucasmccabe/logiqa", split="validation")
        train_dataset = [
            {
                "context": sample["context"].strip(),
                "query": sample["query"].strip(),
                "options": sample["options"],
                "correct_answer": sample["options"][sample["correct_option"]].strip()
            }
            for sample in dataset
        ]
        validation_dataset = train_dataset
    elif dataset_name == "piqa":
        dataset = load_dataset("ybisk/piqa", cache_dir='.')
        train_dataset = [
            {
                "goal": sample["goal"].strip(),
                "options": [sample["sol1"].strip(), sample["sol2"].strip()],
                "correct_answer": str(sample["label"])
            }
            for sample in dataset["validation"]
        ]
        validation_dataset = train_dataset
    elif dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir='.')
        train_dataset = [
            {
                "question": sample["question"].strip(),
                "options": [opt.strip() for opt in sample["choices"]["text"]],
                "correct_answer": str(sample["answerKey"])
            }
            for sample in dataset["test"]
        ]
        validation_dataset = train_dataset

def plot_each_metric_separately(steps, metrics_by_prompt):
    metric_names = list(next(iter(metrics_by_prompt.values())).keys())
    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        for prompt_label, metrics in metrics_by_prompt.items():
            plt.plot(steps, metrics[metric_name], marker='o', label=prompt_label)
        plt.title(f"{metric_name.replace('_', ' ').title()} Across Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend(title="Prompt")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{metric_name}.png')
        plt.close()

def plot_all_metrics_comparison(steps, metrics_by_prompt):
    metric_names = list(next(iter(metrics_by_prompt.values())).keys())
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    for ax, metric_name in zip(axes, metric_names):
        for prompt_label, metrics in metrics_by_prompt.items():
            ax.plot(steps, metrics[metric_name], marker='o', label=prompt_label)
        ax.set_title(f"{metric_name.replace('_', ' ').title()}")
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend(title="Prompt")
    axes[-1].set_xlabel("Training Step")
    plt.tight_layout()
    plt.savefig('all_metrics_comparison.png')
    plt.close()

def plot_all_metrics_together(steps, metrics_by_prompt):
    metric_names = list(next(iter(metrics_by_prompt.values())).keys())
    plt.figure(figsize=(12, 8))
    for metric_name in metric_names:
        for prompt_label, metrics in metrics_by_prompt.items():
            values = np.array(metrics[metric_name])
            if values.max() == values.min():
                normalized_values = np.zeros_like(values)
            else:
                normalized_values = (values - values.min()) / (values.max() - values.min())
            plt.plot(steps, normalized_values, marker='o', label=f"{metric_name.replace('_', ' ').title()}")
    plt.title("All Metrics (Normalized) Across Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Normalized Metric Value")
    plt.legend(title="Metric")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_metrics_together.png')
    plt.close()

def save_metrics_to_file(metrics, step, filename="metrics_per_step.json"):
    with open(filename, 'a') as f:
        json.dump({"step": step, "metrics": metrics}, f, indent=4)
        f.write("\n")

def save_samples_to_file(samples, filename="selected_samples.json"):
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)

def load_samples_from_file(filename="selected_samples.json"):
    with open(filename, 'r') as f:
        return json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_choice = input("Enter dataset name (logiqa, piqa, arc): ").strip()
vars = {
    "dataset": dataset_choice
}

model_size = input("Enter model size (e.g. 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b): ").strip()
model_name_base = f"EleutherAI/pythia-{model_size}-deduped"
cache_base_path = f"./pythia-{model_size}-deduped"
deduped_steps = [
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

steps = []
metrics_by_prompt = {
    "dataset_prompt": {
        "spectral_entropy": [],
        "graph_energy": [],
        "density": []
    }
}

train_dataset, validation_dataset = load_data(vars["dataset"])

loaded_sample_data = load_samples_from_file(filename="selected_samples.json")
for step in deduped_steps:
    print(f"\n\n=== Running Deduplicated Model: {model_name_base} @ {step} ===")
    steps.append(int(step.replace("step", "")))

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_base,
        revision=step,
        cache_dir=f"{cache_base_path}/{step}"
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_base,
        revision=step,
        cache_dir=f"{cache_base_path}/{step}"
    )
    tokenizer.pad_token = tokenizer.eos_token

    step_metrics = {
        "spectral_entropy": [],
        "graph_energy": [],
        "density": []
    }

    for entry in loaded_sample_data:
      if vars["dataset"] == "logiqa":
          context = entry["context"]
          query = entry["query"]
          correct_answer = entry["correct_answer"]
          options = entry["options"]
          prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nOPTIONS:\n"
          for idx, opt in enumerate(options):
              prompt += f"{idx}: {opt}\n"
      elif vars["dataset"] == "piqa":
          goal = entry["goal"]
          options = entry["options"]
          correct_answer = entry["correct_answer"]
          prompt = f"GOAL:\n{goal}\n\nSolution 1:\n{options[0]}\nSolution 2:\n{options[1]}"
      elif vars["dataset"] == "arc":
          question = entry["question"]
          options = entry["options"]
          correct_answer = entry["correct_answer"]
          prompt = f"Question:\n{question}\n\nOption 1:\n{options[0]}\nOption 2:\n{options[1]}\nOption 3:\n{options[2]}\nOption 4:\n{options[3]}"
      else:
          question = entry["question"]
          correct_answer = entry["preferred_answer"]
          prompt = f"{question}"

      G, generated_text = build_word_transition_graph(prompt, model, tokenizer, device, token_limit=1200)
      metrics = get_spectral_metrics_from_graph(G)

      print(f"Step: {step}, Prompt: {entry}")
      for metric_name, value in metrics.items():
          print(f"{metric_name}: {value}")
          step_metrics[metric_name].append(value)

    save_metrics_to_file(step_metrics, step)

    for metric_name in step_metrics:
        avg_value = np.mean(step_metrics[metric_name])
        metrics_by_prompt["dataset_prompt"][metric_name].append(avg_value)

plot_each_metric_separately(steps, metrics_by_prompt)
plot_all_metrics_comparison(steps, metrics_by_prompt)
plot_all_metrics_together(steps, metrics_by_prompt)
