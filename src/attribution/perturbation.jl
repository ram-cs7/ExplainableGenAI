module PerturbationAttribution

using Statistics
using Flux

export compute_deletion_curve, compute_insertion_curve, compute_perturbation_attributions
export get_target_probability

"""
    compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_position)

Compute deletion curve by removing tokens one by one based on importance ordering.
"""
function compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_position)
    original_prob = get_target_probability(model, tokenizer, input_ids, generated_ids, target_position)
    
    probs = Float32[original_prob]
    remaining_ids = copy(input_ids)
    
    while length(remaining_ids) > 0
        # Remove least important token (randomly for now, should be based on importance)
        idx_to_remove = rand(1:length(remaining_ids))
        deleteat!(remaining_ids, idx_to_remove)
        
        if isempty(remaining_ids)
            push!(probs, 0.0f0)
            break
        end
        
        # Compute probability with remaining tokens
        new_prob = get_target_probability(model, tokenizer, remaining_ids, generated_ids, target_position)
        push!(probs, new_prob)
    end
    
    return probs
end

"""
    compute_insertion_curve(model, tokenizer, input_ids, generated_ids, target_position)

Compute insertion curve by adding tokens one by one.
"""
function compute_insertion_curve(model, tokenizer, input_ids, generated_ids, target_position)
    # Start with empty input
    current_ids = Int[]
    probs = Float32[0.0f0]  # Probability with no input tokens
    
    # Random order for insertion (should be based on importance)
    insertion_order = randperm(length(input_ids))
    
    for idx in insertion_order
        push!(current_ids, input_ids[idx])
        
        # Compute probability with current tokens
        new_prob = get_target_probability(model, tokenizer, current_ids, generated_ids, target_position)
        push!(probs, new_prob)
    end
    
    return probs
end

"""
    compute_perturbation_attributions(model, tokenizer, input_ids, generated_ids, target_position; method="deletion")

Compute perturbation-based attributions using deletion or insertion method.
"""
function compute_perturbation_attributions(model, tokenizer, input_ids, generated_ids, target_position; method="deletion")
    if method == "deletion"
        curve = compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_position)
        # Attribution is the drop in probability when each token is removed
        attributions = zeros(Float32, length(input_ids))
        
        for i in 1:length(input_ids)
            # Create input without token i
            masked_ids = deleteat!(copy(input_ids), i)
            
            if isempty(masked_ids)
                attributions[i] = curve[1]  # Original probability
                continue
            end
            
            # Compute probability without token i
            masked_prob = get_target_probability(model, tokenizer, masked_ids, generated_ids, target_position)
            attributions[i] = curve[1] - masked_prob
        end
    elseif method == "insertion"
        curve = compute_insertion_curve(model, tokenizer, input_ids, generated_ids, target_position)
        # Attribution is the increase in probability when each token is added
        attributions = zeros(Float32, length(input_ids))
        
        for i in 1:length(input_ids)
            # Create input with only token i
            single_token_ids = [input_ids[i]]
            
            # Compute probability with only token i
            single_prob = get_target_probability(model, tokenizer, single_token_ids, generated_ids, target_position)
            attributions[i] = single_prob - curve[1]  # Difference from baseline (no tokens)
        end
    else
        println("Warning: Unknown perturbation method $method. Falling back to :deletion")
        return compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_pos)
    end
    
    return attributions
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

end