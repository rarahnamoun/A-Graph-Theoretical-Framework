import torch
import json
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7"
]

def load_prompts_from_json(file_path="arc_samples.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    prompts = [sample["prompt"] for sample in data["samples"]]
    references = [sample["references"] for sample in data["samples"]]
    return prompts, references

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    return tokenizer, model

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

def build_word_transition_graph(prompt, model, tokenizer, token_limit=1500):
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
        top_probs, top_indices = torch.topk(probs, 90)
        selected_token_id = generated_ids[input_length + i]
        prev_token_id = inputs.input_ids[0, -1] if i == 0 else generated_ids[input_length + i - 1]
        prev_word = tokenizer.decode(prev_token_id).strip()
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            candidate_word = tokenizer.decode(token_id).strip()
            G.add_node(prev_word)
            G.add_node(candidate_word)
            G.add_edge(prev_word, candidate_word, weight=prob)
    return G, generated_text

def get_spectral_metrics_from_graph(G):
    if len(G) == 0:
        return {
            "spectral_entropy": 0.0,
            "graph_energy": 0.0,
            "density": 0.0
        }
    return {
        "spectral_entropy": spectral_entropy(G),
        "graph_energy": graph_energy(G),
        "density": graph_density(G)
    }

def plot_metrics_bar_chart(all_metrics, output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
    metrics_names = list(next(iter(all_metrics.values())).keys())
    for metric in metrics_names:
        plt.figure(figsize=(12, 6))
        values = [all_metrics[model][metric] for model in all_metrics]
        plt.bar(all_metrics.keys(), values)
        plt.title(f'{metric} per Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric}_bar_chart.png"))
        plt.close()

def sanitize_model_name(model_path: str) -> str:
    return model_path.replace("/", "_")

def main():
    os.makedirs("model_results", exist_ok=True)
    all_model_metrics = {}
    PROMPTS, REFERENCES = load_prompts_from_json("arc_samples.json")

    for model_path in MODEL_LIST:
        print(f"Running model: {model_path}")
        try:
            tokenizer, model = load_model(model_path)
            avg_metrics = {k: 0.0 for k in [
                "spectral_entropy", "graph_energy", "density"
            ]}
            model_outputs = {}
            for idx, prompt in enumerate(PROMPTS):
                G, generated_text = build_word_transition_graph(prompt, model, tokenizer)
                metrics = get_spectral_metrics_from_graph(G)
                model_outputs[prompt] = {
                    "generated": generated_text,
                    "reference": REFERENCES[idx]
                }
                for k in avg_metrics:
                    avg_metrics[k] += metrics.get(k, 0.0)

            for k in avg_metrics:
                avg_metrics[k] /= len(PROMPTS)
            all_model_metrics[model_path] = avg_metrics

            model_name_sanitized = sanitize_model_name(model_path)
            with open(f"model_results/metrics_{model_name_sanitized}.json", "w") as f:
                json.dump(avg_metrics, f, indent=2)
            with open(f"model_results/outputs_{model_name_sanitized}.json", "w") as f:
                json.dump(model_outputs, f, indent=2)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing model {model_path}: {e}")

    with open("model_metrics.json", "w") as f:
        json.dump(all_model_metrics, f, indent=2)

    plot_metrics_bar_chart(all_model_metrics)

main()
