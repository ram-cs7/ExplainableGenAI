# ExplainableGenAI.jl

**A native Julia infrastructure for Mechanistic Interpretability and Agentic Traceability.**

[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Ready-brightgreen.svg)]()

---

## 🌟 Overview

`ExplainableGenAI.jl` provides a unified computational bridge between Large Language Models and the Julia Scientific Machine Learning (SciML) ecosystem. By natively implementing modern interpretability architectures (e.g., Sparse Autoencoders) in `Flux.jl` and `Zygote.jl`, it enables researchers to apply these techniques directly to scientific models—such as physics-informed neural networks or differential equation solvers—without the cross-language serialization friction associated with Python-based frameworks like `TransformerLens`.

The framework currently focuses on three core areas derived from 2025–2026 methodologies:

1. 🧠 **Mechanistic Interpretability (`MechanisticInterpretability`)**
   - Native Julia implementations of modern SAE architectures: **TopK, BatchTopK, and JumpReLU SAEs** with auxiliary dead-feature tracking.
   - **Equivariant Sparse Autoencoders (E-SAEs)** based on Erdogan & Lucic (2025), designed for scientific data with group invariances.
   
2. ⚖️ **CoT Interventional Faithfulness (`FaithfulnessEvaluation`)**
   - Log-likelihood KL-divergence metrics to determine if a model causally relies on its Chain-of-Thought (CoT).
   - Designed around the Faithful Reasoning via Intervention Training (**FRIT**, 2025) paradigm.

3. 🤖 **Agentic Traceability Engine (`AgenticTraceability`)**
   - Temporal semantic lineage tracking via Directed Acyclic Graphs (`AgentTrace`).
   - Implements **AgenTracer**-style (2025) silent failure diagnosis through programmatic **Counterfactual Replay**, mathematically isolating faults to specific agent modules (Memory, Planning, Tool Execution).

*Note: The LLM hooking utilities currently rely on `Transformers.jl` and are primarily validated on the GPT-2 family of models. Expanding native compatibility to frontier-scale architectures requires extending `Transformers.jl` layer accessors.*

*Scope Note: While the core focus of this framework is on modern Mechanistic Interpretability, CoT Faithfulness, and Agentic Traceability (which form the tested core of our JOSS submission), the repository also ships several classical baseline attribution methods (e.g., Integrated Gradients, Attention Rollout) and a visualization layer. These are provided strictly for comparative baseline purposes and fall outside the primary theoretical scope.*

---

## 🚀 Quick Start Demo

Run the 2026 SOTA demonstration script to see all three pillars in action:

```bash
julia --project=. examples/2026_sota_demo.jl
```

### 1. Training a Native TopK SAE on Real GPT-2 Activations
```julia
using ExplainableGenAI
using ExplainableGenAI.LLMWrapper

# Load real HuggingFace model and hook into its forward pass
model, tokenizer = load_model("gpt2")
ht = HookedTransformer(model)

input_ids = tokenizer("The future of AI is promising").token
logits, cache = cache_activations(ht, input_ids)

# Extract genuine hidden states from the deepest layer
hidden_states = cache["layer_11"]
d_model = size(hidden_states, 1)

# Initialize and train TopK SAE for extracting monosemantic features
sae = SparseAutoencoder(d_model, d_model * 4, k=16)
train_sae!(sae, reshape(hidden_states, d_model, :); epochs=10, lr=1e-3)
```

### 2. Measuring CoT Faithfulness
```julia
# Corrupt the reasoning trace and measure if the model changes its answer
kl_div = logit_kl_divergence(clean_logits, corrupted_logits)

if kl_div > 1.0
    println("Faithful: Corrupting the reasoning trace changed the final answer.")
else
    println("Unfaithful: The model ignored the corrupted reasoning.")
end
```

### 3. Agentic Fault Diagnosis
```julia
# Diagnose a failed multi-agent workflow
trace = AgentTrace("session_9942", [...nodes...], :Failure)

# Uses counterfactual replay to assign error attribution scores
scores = diagnose_failure(trace)
println(scores[:Planning]) # e.g., 0.82 (Planning module caused the failure)
```

---

## 📦 Installation

```julia
# Clone the repository
cd("ExplainableGenAI")

# Activate and instantiate the environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## 🔬 Research & Citations

This package implements methodologies from recent top-tier AI conferences:

1. **TopK Sparse Autoencoders:** "Scaling and evaluating sparse autoencoders" (Gao et al., 2024) & "BatchTopK Sparse Autoencoders" (Bussmann, Leask & Nanda, 2024).
2. **CoT Faithfulness Protocols:** "FaithCoT-Bench" & "Faithful Reasoning via Intervention Training (FRIT)" (2025).
3. **Agentic Counterfactual Replay:** "AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?" (2025).

### Citing This Work

```bibtex
@software{explainablegenai2026,
  title={ExplainableGenAI: A SOTA Framework for Mechanistic Interpretability and Agentic Traceability},
  author={Sairam Chennaka},
  year={2026},
  url={https://github.com/ram-cs7/ExplainableGenAI}
}
```

---

## 📁 Project Structure

```text
ExplainableGenAI/
├── src/
│   ├── ExplainableGenAI.jl              # Main exported module & unified explain() API
│   ├── models/
│   │   ├── llm_wrapper.jl               # HuggingFace model loading via Transformers.jl
│   │   └── img_wrapper.jl               # Image model loading & Grad-CAM
│   ├── attribution/
│   │   ├── mechanistic.jl               # TopK / BatchTopK / JumpReLU SAEs, E-SAEs, Activation Hooking
│   │   ├── agentic.jl                   # Agentic Traceability, Counterfactual Replay, [Experimental] Collusion Detection
│   │   ├── autoresearch.jl              # [Experimental] Automated Interpretability Agent (feature discovery)
│   │   ├── gradient.jl                  # [Baseline] Grad×Input, Integrated Gradients, SmoothGrad, BlurIG, Expected Gradients
│   │   ├── attention.jl                 # [Baseline] Attention Rollout, Attention Flow, ALTI
│   │   └── perturbation.jl              # [Baseline] Deletion curves, Insertion curves
│   ├── evaluation/
│   │   ├── faithfulness.jl              # Logit KL-Divergence & FRIT CoT Faithfulness Protocol
│   │   ├── metrics.jl                   # [Baseline] AUC, Comprehensiveness, Sufficiency, Infidelity, Sensitivity
│   │   └── experiments.jl               # SQuAD dataset loading & QA experiment runner
│   └── visualization/
│       ├── plots.jl                     # [Baseline] CairoMakie token heatmaps, deletion curves, Grad-CAM overlays
│       └── dashboard.jl                 # [Baseline] Interactive Pluto.jl dashboard
├── examples/
│   ├── 2026_sota_demo.jl                # End-to-end 2026 SOTA demo (real GPT-2)
│   ├── ablation_study.jl                # IG steps & attention discard ratio ablation
│   ├── analyze_results.jl               # Results analysis & plotting
│   ├── run_experiments.jl               # Batch experiment runner
│   └── test_demo.jl                     # Quick test script
├── test/
│   └── runtests.jl                      # Unit tests (SAE shape/sparsity, KL math, agentic attribution)
├── Project.toml                         # Julia dependencies
└── README.md                            # This file
```

---

## 📝 License

This project is licensed under the MIT License.
