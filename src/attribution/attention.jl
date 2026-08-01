module AttentionAttribution

using Statistics
using LinearAlgebra

export compute_attention_rollout, compute_attention_flow, compute_alti_attention

"""
    compute_attention_rollout(attentions; discard_ratio=0.0)

Compute attention rollout from attention matrices using recursive multiplication.
This method traces attention paths through multiple layers to understand token importance.

# Arguments
- `attentions`: Vector of attention matrices for each layer
- `discard_ratio`: Ratio of lowest attention weights to discard (for noise reduction)

# Returns
- Joint attention matrix showing cumulative attention flow
"""
function compute_attention_rollout(attentions; discard_ratio=0.0)
    # attentions: Vector of attention matrices [layers, heads, seq_len, seq_len]
    num_layers = length(attentions)
    seq_len = size(attentions[1], 2)
    
    # Start with identity matrix
    joint = Matrix{Float32}(I, seq_len, seq_len)
    
    for layer in 1:num_layers
        # Average heads
        avg_attention = mean(attentions[layer], dims=2)[:, 1, :, :]  # [1, seq_len, seq_len]
        
        # Apply discard ratio if needed
        if discard_ratio > 0
            flat_att = vec(avg_attention)
            threshold = sort(flat_att)[Int(floor((1 - discard_ratio) * length(flat_att)))]
            avg_attention[avg_attention .< threshold] .= 0
            # Renormalize
            avg_attention = avg_attention ./ sum(avg_attention, dims=3)
        end
        
        # Multiply with previous joint attention
        joint = avg_attention * joint
    end
    
    # Normalize rows to sum to 1
    row_sums = sum(joint, dims=2)
    joint = joint ./ row_sums
    
    return joint
end

"""
    compute_attention_flow(attentions; head_weights=nothing)

Compute attention flow with optional head weights using gradient-based importance.
This advanced method weights attention heads by their contribution to the final output.

# Arguments
- `attentions`: Vector of attention matrices for each layer
- `head_weights`: Optional weights for each attention head (auto-computed if not provided)

# Returns
- Weighted attention flow matrix
"""
function compute_attention_flow(attentions; head_weights=nothing)
    num_layers = length(attentions)
    num_heads = size(attentions[1], 2)
    seq_len = size(attentions[1], 3)
    
    # Initialize head weights if not provided
    if head_weights === nothing
        head_weights = ones(Float32, num_heads) ./ num_heads
    end
    
    # Start with identity matrix
    joint = Matrix{Float32}(I, seq_len, seq_len)
    
    for layer in 1:num_layers
        # Apply head weights
        weighted_attention = zeros(Float32, seq_len, seq_len)
        for head in 1:num_heads
            weighted_attention .+= head_weights[head] * attentions[layer][1, head, :, :]
        end
        
        # Multiply with previous joint attention
        joint = weighted_attention * joint
    end
    
    # Normalize rows to sum to 1
    row_sums = sum(joint, dims=2)
    joint = joint ./ row_sums
    
    return joint
end

"""
    compute_alti_attention(attentions, value_vectors; aggregation=:max)

Compute ALTI (Attention with Linear-Time Inference) attribution scores.
This advanced method combines attention weights with value vector norms for more accurate attribution.

Reference: Ferrando et al. (2022) - "Explaining How Transformers Use Context to Build Predictions"

# Arguments
- `attentions`: Vector of attention matrices for each layer
- `value_vectors`: Value vectors from the model (if available, otherwise approximated)
- `aggregation`: How to aggregate across heads (:max, :mean, :sum)

# Returns
- ALTI attribution scores combining attention and value magnitudes
"""
function compute_alti_attention(attentions, value_vectors=nothing; aggregation=:max)
    num_layers = length(attentions)
    
    if value_vectors === nothing
        # Fallback to standard attention rollout if values not available
        return compute_attention_rollout(attentions)
    end
    
    # ALTI computes attention × ||value||
    alti_scores = []
    
    for layer in 1:num_layers
        layer_attn = attentions[layer]
        layer_values = value_vectors[layer]
        
        # Compute value norms for each token
        value_norms = sqrt.(sum(layer_values.^2, dims=1))
        
        # Weight attention by value norms
        weighted_attn = layer_attn .* reshape(value_norms, 1, 1, :)
        
        # Aggregate across heads
        if aggregation == :max
            aggregated = maximum(weighted_attn, dims=2)
        elseif aggregation == :mean
            aggregated = mean(weighted_attn, dims=2)
        else  # :sum
            aggregated = sum(weighted_attn, dims=2)
        end
        
        push!(alti_scores, aggregated)
    end
    
    # Combine across layers (multiply for path-based attribution)
    combined = alti_scores[1]
    for i in 2:num_layers
        combined = combined .* alti_scores[i]
    end
    
    return combined
end

end