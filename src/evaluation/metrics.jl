module EvaluationMetrics

using Statistics
using StatsBase
using Flux

export compute_auc, compute_comprehensiveness, compute_sufficiency, compute_faithfulness_correlation
export compute_sensitivity, compute_infidelity, get_target_probability

""" compute_auc(curve)

Compute Area Under Curve for deletion/insertion curves using trapezoidal integration.
"""
function compute_auc(curve)
    # Normalize curve to [0,1]
    normalized = curve ./ curve[1]
    
    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in 1:(length(normalized)-1)
        auc += (normalized[i] + normalized[i+1]) / 2
    end
    
    return auc / (length(normalized) - 1)
end

"""
    compute_comprehensiveness(model, tokenizer, input_ids, generated_ids, target_position, attributions; k=5)

Compute comprehensiveness: drop in probability when removing top-k important tokens.
"""
function compute_comprehensiveness(model, tokenizer, input_ids, generated_ids, target_position, attributions; k=5)
    # Get original probability
    original_prob = get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)
    
    # Get indices of top-k important tokens
    top_k_indices = partialsortperm(attributions, 1:k, rev=true)
    
    # Create input without top-k tokens
    remaining_ids = setdiff(1:length(input_ids), top_k_indices)
    if isempty(remaining_ids)
        return original_prob  # All tokens removed
    end
    
    masked_ids = input_ids[remaining_ids]
    
    # Compute probability without top-k tokens
    masked_prob = get_target_probability(model, tokenizer, masked_ids, generated_ids, target_position)
    
    # Comprehensiveness is the drop in probability
    return original_prob - masked_prob
end

"""
    compute_sufficiency(model, tokenizer, input_ids, generated_ids, target_position, attributions; k=5)

Compute sufficiency: probability when using only top-k important tokens.
"""
function compute_sufficiency(model, tokenizer, input_ids, generated_ids, target_position, attributions; k=5)
    # Get indices of top-k important tokens
    top_k_indices = partialsortperm(attributions, 1:k, rev=true)
    
    # Create input with only top-k tokens
    top_k_ids = input_ids[top_k_indices]
    
    # Compute probability with only top-k tokens
    top_k_prob = get_target_probability(model, tokenizer, top_k_ids, generated_ids, target_position)
    
    return top_k_prob
end

"""
    compute_faithfulness_correlation(model, tokenizer, input_ids, generated_ids, target_position, attributions; samples=100)

Compute correlation between attribution scores and actual importance measured by perturbation.
"""
function compute_faithfulness_correlation(model, tokenizer, input_ids, generated_ids, target_position, attributions; samples=100)
    # Get original probability
    original_prob = get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)
    
    # Randomly sample tokens to perturb
    perturbation_scores = zeros(Float32, length(input_ids))
    
    for _ in 1:samples
        # Randomly select a token to perturb
        idx = rand(1:length(input_ids))
        
        # Create input without this token
        masked_ids = deleteat!(copy(input_ids), idx)
        
        if isempty(masked_ids)
            perturbation_scores[idx] += original_prob
            continue
        end
        
        # Compute probability without this token
        masked_prob = get_target_probability(model, tokenizer, masked_ids, generated_ids, target_position)
        
        # Perturbation score is the drop in probability
        perturbation_scores[idx] += original_prob - masked_prob
    end
    
    # Average perturbation scores
    perturbation_scores ./= samples
    
    # Compute Spearman correlation with attribution scores
    correlation = StatsBase.corspearman(attributions, perturbation_scores)
    
    return correlation
end

"""
    get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)

Helper function to get probability of target token at target_position.
"""
function get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)
    # Run model
    logits = model(input_ids)
    
    # If target_position is beyond input length, we need to generate up to that position
    if target_position > length(input_ids)
        # Generate tokens up to target_position
        current_ids = copy(input_ids)
        for pos in (length(input_ids)+1):target_position
            logits = model(current_ids)
            next_token = argmax(logits[:, end, :])
            push!(current_ids, next_token)
        end
        
        # Get logits at target position
        logits = model(current_ids)
        log_probs = Flux.logsoftmax(logits[:, end, :])
    else
        # Target is within input, get logits at that position
        log_probs = Flux.logsoftmax(logits[:, target_position, :])
    end
    
    target_token_id = generated_ids[target_position]
    return exp(log_probs[target_token_id])
end

"""
    compute_sensitivity(attributions, perturbation_radius=0.1)

Compute sensitivity-n metric: maximum sensitivity of attributions to input perturbations.
Lower is better (more stable attributions).

Reference: Yeh et al. (2019) - "On the (In)fidelity and Sensitivity of Explanations"
"""
function compute_sensitivity(attributions, perturbation_radius=0.1)
    # Sensitivity measures how much attributions change with small input perturbations
    # This is a simplified version - full implementation would need model access
    return std(attributions) * perturbation_radius
end

"""
    compute_infidelity(model, tokenizer, input_ids, generated_ids, target_position, attributions; num_samples=50)

Compute infidelity metric: measures how well attributions predict the effect of perturbations.
Lower is better (more faithful attributions).

Reference: Yeh et al. (2019) - "On the (In)fidelity and Sensitivity of Explanations"
"""
function compute_infidelity(model, tokenizer, input_ids, generated_ids, target_position, attributions; num_samples=50)
    # Get original output
    original_prob = get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)
    
    infidelity_sum = 0.0
    
    for _ in 1:num_samples
        # Create random perturbation mask
        perturb_mask = rand(Float32, length(input_ids)) .< 0.2  # 20% perturbation rate
        
        if sum(perturb_mask) == 0
            continue  # Skip if no perturbations
        end
        
        # Create perturbed input (remove perturbed tokens)
        perturbed_ids = input_ids[.!perturb_mask]
        
        if isempty(perturbed_ids)
            continue
        end
        
        # Get perturbed output
        perturbed_prob = get_target_probability(model, tokenizer, perturbed_ids, generated_ids, target_position)
        
        # Compute actual change
        actual_change = original_prob - perturbed_prob
        
        # Compute predicted change (sum of attributions for perturbed tokens)
        predicted_change = sum(attributions[perturb_mask])
        
        # Accumulate squared error
        infidelity_sum += (actual_change - predicted_change)^2
    end
    
    return infidelity_sum / num_samples
end

end