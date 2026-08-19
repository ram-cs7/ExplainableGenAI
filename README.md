# ExplainableGenAI.jl

**A Native Julia Framework for Mechanistic Interpretability and Agentic Traceability in Scientific Machine Learning.**

[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-NeurIPS%202026%20Workshop%20Submission-brightgreen.svg)]()

---

## 🌟 Overview

`ExplainableGenAI.jl` provides a unified computational bridge between Large Language Models and the Julia Scientific Machine Learning (SciML) ecosystem. By natively implementing modern interpretability architectures (e.g., Sparse Autoencoders) in `Flux.jl` and `Zygote.jl`, it enables researchers to apply these techniques directly to scientific models—such as Physics-Informed Neural Networks (PINNs) or Neural Ordinary Differential Equations—without the cross-language serialization friction associated with Python-based frameworks.

This framework strictly aligns with the dual-focus of our NeurIPS 2026 Interpretability for Discovery Workshop submission:

1. 🧠 **Mechanistic Interpretability via Zygote Adjoints**
   - Native Julia implementations of modern SAE architectures: **TopK, BatchTopK, and JumpReLU SAEs**.
   - Custom `ChainRulesCore.jl` pullbacks (Lemma 1) that resolve the discontinuous thresholding problem via Gaussian surrogate approximations.
   - **Equivariant Sparse Autoencoders (E-SAEs)** designed for scientific data with physical group invariances.
   
2. 🤖 **Agentic Traceability via Causal DAGs**
   - Temporal semantic lineage tracking for multi-agent systems via Directed Acyclic Graphs (`AgentTrace`).
   - Implements **O(log D) Binary Causal Search** via counterfactual replay, mathematically isolating silent faults to specific agent modules in linear execution chains with minimal intervention overhead.

---

## 🚀 Quick Start

### 1. Training a Native JumpReLU SAE (Lemma 1)
```julia
using ExplainableGenAI
using ExplainableGenAI.LLMWrapper

# Hook into a model's forward pass
model, tokenizer = load_model("gpt2")
ht = HookedTransformer(model)

input_ids = tokenizer("The physics simulation is stable.").token
logits, cache = cache_activations(ht, input_ids)

# Extract hidden states
hidden_states = cache["layer_11"]
d_model = size(hidden_states, 1)

# Initialize JumpReLU SAE. 
# Uses custom ChainRulesCore.rrule for discontinuous thresholding!
sae = JumpReLUSparseAutoencoder(d_model, d_model * 4, initial_threshold=0.1f0)
train_sae!(sae, reshape(hidden_states, d_model, :); epochs=10, lr=1e-3)
```

### 2. O(log D) Agentic Fault Diagnosis
```julia
using ExplainableGenAI.AgenticTraceability

# Diagnose a failed multi-agent workflow
trace = AgentTrace("session_9942", [...nodes...], :Failure)

# Uses O(log D) binary causal search to assign absolute error attribution
# Intervenes at the midpoint of the DAG and checks terminal output
scores = diagnose_failure(trace; agent_executor=my_agent_executor_function)
println(scores[:Retriever]) # e.g., 1.0 (Retriever node induced the failure)
```

---

## 🔬 Theoretical Implementation

This package directly implements the mathematical derivations provided in our manuscript:

1. **Lemma 1 (Bounded Convergence of the Adjoint):** Found in `src/attribution/mechanistic.jl`. The custom `ChainRulesCore.rrule` for `jumprelu_activation` uses a Gaussian mollifier to approximate the Dirac delta gradient of the Heaviside step function.
2. **Causal Trace Complexity Analysis:** Found in `src/attribution/agentic.jl`. The `diagnose_failure` routine utilizes a binary search (`O(log D)`) across strict linear chains within the causal graph, isolating root causes exponentially faster than linear scanning.

## 📁 Project Structure

```text
ExplainableGenAI/
├── src/
│   ├── ExplainableGenAI.jl              # Main exported module
│   ├── models/
│   │   ├── llm_wrapper.jl               # HuggingFace model loading via Transformers.jl
│   │   └── img_wrapper.jl               # Vision model loading interfaces
│   ├── attribution/
│   │   ├── mechanistic.jl               # SAEs, Custom Zygote Adjoints (Lemma 1)
│   │   ├── agentic.jl                   # DAG Traceability, O(log D) Binary Causal Search
│   │   ├── gradient.jl                  # [Baseline] Integrated Gradients
│   │   ├── attention.jl                 # [Baseline] Attention Rollout
│   │   ├── perturbation.jl              # [Baseline] Input feature ablation
│   │   └── autoresearch.jl              # Automated feature discovery tooling
│   └── visualization/
│       ├── plots.jl                     # [Baseline] Visualization utilities
│       └── dashboard.jl                 # Interactive visualization dashboards
├── examples/
│   ├── case_study_1_math.jl             # Case study 1: Math-related phenomena
│   └── case_study_2_vision.jl           # Case study 2: Vision-related phenomena
├── test/
│   └── runtests.jl                      # Unit tests (JumpReLU Adjoints, DAG scaling)
├── benchmark_sae.jl                     # Multi-threaded Julia benchmarking script
├── benchmark_torch.py                   # PyTorch hardware baseline script
├── Project.toml                         # Julia dependencies
└── README.md                            # This file
```

---

## 📝 License

This project is licensed under the MIT License.
