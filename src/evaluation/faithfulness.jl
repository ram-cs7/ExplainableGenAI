module FaithfulnessEvaluation

using Flux
using Statistics
using LinearAlgebra

export compute_intervention_faithfulness, logit_kl_divergence, evaluate_cot_faithfulness

# ==============================================================================
# CoT Interventional Faithfulness (ICLR 2026 SOTA)
# ==============================================================================

"""
    logit_kl_divergence(logits_clean, logits_corrupt)

Computes the KL divergence between the output probability distributions 
of the clean trace and the corrupted trace.
"""
function logit_kl_divergence(logits_clean::AbstractVector, logits_corrupt::AbstractVector)
    # Convert logits to probabilities
    p_clean = Flux.softmax(logits_clean)
    p_corrupt = Flux.softmax(logits_corrupt)
    
    # KL(P || Q) = sum(P * log(P / Q))
    # Add epsilon for numerical stability
    eps = 1e-10
    kl = sum(p_clean .* log.((p_clean .+ eps) ./ (p_corrupt .+ eps)))
    return kl
end

"""
    compute_intervention_faithfulness(model, tokenizer, prompt, original_cot, corrupted_cot)

Implements the interventional faithfulness protocol (Lanham et al., 2023).
1. Computes the final answer log-probabilities given `prompt + original_cot`
2. Computes the final answer log-probabilities given `prompt + corrupted_cot`
3. The faithfulness score is the divergence. If KL is near 0, the CoT was NOT faithful 
   (the model ignored the corrupted reasoning and answered anyway).
"""
function compute_intervention_faithfulness(model, tokenizer, prompt::String, original_cot::String, corrupted_cot::String)
    # 1. Base trace
    clean_text = prompt * "\n" * original_cot
    clean_tokens = tokenizer(clean_text).token
    
    # Extract the logits for the final answer prediction (end of sequence)
    logits_clean, _ = model(clean_tokens)
    logits_clean_final = logits_clean[:, end, 1] 
    
    # 2. Corrupted trace
    corrupt_text = prompt * "\n" * corrupted_cot
    corrupt_tokens = tokenizer(corrupt_text).token
    
    logits_corrupt, _ = model(corrupt_tokens)
    logits_corrupt_final = logits_corrupt[:, end, 1]
    
    # 3. Calculate divergence
    kl_div = logit_kl_divergence(logits_clean_final, logits_corrupt_final)
    
    # High KL divergence -> The CoT mattered -> High Faithfulness
    return kl_div
end

"""
    evaluate_cot_faithfulness(dataset_path::String)

Runs the interventional faithfulness evaluation across a dataset of CoT traces.
"""
function evaluate_cot_faithfulness(dataset_path::String)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found at $dataset_path. Provide a valid JSON/CSV dataset of CoT traces."))
    end
    println("Loading CoT faithfulness dataset from $dataset_path...")
    # Users must implement specific JSON/CSV parsing depending on the dataset schema.
    throw(ErrorException("Dataset parser not implemented. Please implement the JSON/CSV schema parsing for your specific faithfulness evaluation dataset."))
end

end # module FaithfulnessEvaluation
