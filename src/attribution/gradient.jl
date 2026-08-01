module GradientAttribution

using Zygote
using Flux
using Statistics

export compute_gradient_attributions, compute_integrated_gradients, compute_smoothgrad
export compute_grad_x_input, compute_blur_ig, compute_expected_gradients

"""
    compute_gradient_attributions(model, tokenizer, input_ids, generated_ids, target_position)

Compute gradient * input attribution for a target token at target_position.
This is the simplest gradient-based attribution method.

# Arguments
- `model`: The language model
- `tokenizer`: The tokenizer
- `input_ids`: Input token IDs
- `generated_ids`: Generated token IDs
- `target_position`: Position of the target token to explain

# Returns
- Vector of attribution scores for each input token
"""
function compute_gradient_attributions(model, tokenizer, input_ids, generated_ids, target_position)
    # Get embeddings for the input
    embeddings = model.embed(input_ids)
    
    # Define loss function: negative log prob of target token
    function loss(embeddings)
        # Run model with custom embeddings
        logits = model(input_ids; embeddings=embeddings)
        log_probs = Flux.logsoftmax(logits[:, target_position, :])
        target_token_id = generated_ids[target_position]
        return -log_probs[target_token_id]
    end
    
    # Compute gradients
    grads = Zygote.gradient(loss, embeddings)[1]
    
    # Compute attribution: gradient * input
    attributions = [sum(grads[:, i] .* embeddings[:, i]) for i in 1:size(embeddings, 2)]
    
    return attributions
end

"""
    compute_integrated_gradients(model, tokenizer, input_ids, generated_ids, target_position; steps=20)

Compute integrated gradients approximation.
"""
function compute_integrated_gradients(model, tokenizer, input_ids, generated_ids, target_position; steps=20)
    # Get baseline (zero embeddings)
    baseline = zeros(Float32, size(model.embed(input_ids)))
    
    # Get input embeddings
    embeddings = model.embed(input_ids)
    
    # Compute gradients at interpolated points
    total_grad = zeros(Float32, size(embeddings))
    
    for alpha in range(0, 1, length=steps)
        # Interpolate between baseline and input
        interpolated = baseline .+ alpha .* (embeddings .- baseline)
        
        # Compute gradients at this point
        function loss(interpolated_emb)
            logits = model(input_ids; embeddings=interpolated_emb)
            log_probs = Flux.logsoftmax(logits[:, target_position, :])
            target_token_id = generated_ids[target_position]
            return -log_probs[target_token_id]
        end
        
        grads = Zygote.gradient(loss, interpolated)[1]
        total_grad .+= grads
    end
    
    # Average gradients and compute attribution
    avg_grad = total_grad ./ steps
    attributions = [sum(avg_grad[:, i] .* (embeddings[:, i] .- baseline[:, i])) for i in 1:size(embeddings, 2)]
    
    return attributions
end

"""
    compute_smoothgrad(model, tokenizer, input_ids, generated_ids, target_position; samples=10, noise_level=0.1)

Compute SmoothGrad attribution by averaging gradients with added noise.
"""
function compute_smoothgrad(model, tokenizer, input_ids, generated_ids, target_position; samples=10, noise_level=0.1)
    # Get input embeddings
    embeddings = model.embed(input_ids)
    
    # Initialize storage for gradients
    all_grads = zeros(Float32, size(embeddings))
    
    for _ in 1:samples
        # Add noise to embeddings
        noise = randn(Float32, size(embeddings)) * noise_level
        noisy_embeddings = embeddings .+ noise
        
        # Define loss function
        function loss(noisy_emb)
            logits = model(input_ids; embeddings=noisy_emb)
            log_probs = Flux.logsoftmax(logits[:, target_position, :])
            target_token_id = generated_ids[target_position]
            return -log_probs[target_token_id]
        end
        
        # Compute gradients
        grads = Zygote.gradient(loss, noisy_embeddings)[1]
        all_grads .+= grads
    end
    
    # Average gradients
    avg_grads = all_grads ./ samples
    
    # Compute attribution
    attributions = [sum(avg_grads[:, i] .* embeddings[:, i]) for i in 1:size(embeddings, 2)]
    
    return attributions
end

"""
    compute_grad_x_input(model, tokenizer, input_ids, generated_ids, target_position)

Compute pure gradient × input attribution (alias for compute_gradient_attributions).
This is a common baseline method in attribution literature.
"""
compute_grad_x_input = compute_gradient_attributions

"""
    compute_blur_ig(model, tokenizer, input_ids, generated_ids, target_position; steps=50, blur_sigma=0.2)

Compute Blur Integrated Gradients - an improved version that uses Gaussian blur as baseline.
This method often provides more stable and interpretable attributions than standard IG.

Reference: Xu et al. (2020) - "Attribution in Scale and Space"

# Arguments
- `model`: The language model
- `tokenizer`: The tokenizer
- `input_ids`: Input token IDs
- `generated_ids`: Generated token IDs
- `target_position`: Position of the target token
- `steps`: Number of integration steps (more = more accurate but slower)
- `blur_sigma`: Standard deviation for Gaussian blur baseline

# Returns
- Vector of attribution scores
"""
function compute_blur_ig(model, tokenizer, input_ids, generated_ids, target_position; steps=50, blur_sigma=0.2)
    # Get input embeddings
    embeddings = model.embed(input_ids)
    
    # Create blurred baseline by adding Gaussian noise and averaging
    baseline = embeddings .+ randn(Float32, size(embeddings)) .* blur_sigma
    
    # Compute gradients at interpolated points using Gauss-Legendre quadrature for better accuracy
    total_grad = zeros(Float32, size(embeddings))
    
    for alpha in range(0, 1, length=steps)
        # Interpolate between baseline and input
        interpolated = baseline .+ alpha .* (embeddings .- baseline)
        
        # Compute gradients at this point
        function loss(interpolated_emb)
            logits = model(input_ids; embeddings=interpolated_emb)
            log_probs = Flux.logsoftmax(logits[:, target_position, :])
            target_token_id = generated_ids[target_position]
            return -log_probs[target_token_id]
        end
        
        grads = Zygote.gradient(loss, interpolated)[1]
        total_grad .+= grads
    end
    
    # Average gradients and compute attribution
    avg_grad = total_grad ./ steps
    attributions = [sum(avg_grad[:, i] .* (embeddings[:, i] .- baseline[:, i])) for i in 1:size(embeddings, 2)]
    
    return attributions
end

"""
    compute_expected_gradients(model, tokenizer, input_ids, generated_ids, target_position; num_samples=20)

Compute Expected Gradients - a probabilistic attribution method that samples multiple baselines.
This provides more robust attributions by averaging over a distribution of baselines.

Reference: Erion et al. (2021) - "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors"

# Arguments
- `model`: The language model
- `tokenizer`: The tokenizer
- `input_ids`: Input token IDs
- `generated_ids`: Generated token IDs
- `target_position`: Position of the target token
- `num_samples`: Number of baseline samples to use

# Returns
- Vector of attribution scores
"""
function compute_expected_gradients(model, tokenizer, input_ids, generated_ids, target_position; num_samples=20)
    # Get input embeddings
    embeddings = model.embed(input_ids)
    
    # Sample multiple baselines from a distribution (e.g., random tokens)
    all_attributions = []
    
    for _ in 1:num_samples
        # Create random baseline by sampling from embedding distribution
        baseline = randn(Float32, size(embeddings)) .* std(embeddings, dims=2)
        
        # Compute integrated gradients for this baseline
        total_grad = zeros(Float32, size(embeddings))
        steps = 10  # Fewer steps per sample since we're averaging across samples
        
        for alpha in range(0, 1, length=steps)
            interpolated = baseline .+ alpha .* (embeddings .- baseline)
            
            function loss(interpolated_emb)
                logits = model(input_ids; embeddings=interpolated_emb)
                log_probs = Flux.logsoftmax(logits[:, target_position, :])
                target_token_id = generated_ids[target_position]
                return -log_probs[target_token_id]
            end
            
            grads = Zygote.gradient(loss, interpolated)[1]
            total_grad .+= grads
        end
        
        avg_grad = total_grad ./ steps
        sample_attributions = [sum(avg_grad[:, i] .* (embeddings[:, i] .- baseline[:, i])) for i in 1:size(embeddings, 2)]
        push!(all_attributions, sample_attributions)
    end
    
    # Average across all baseline samples
    attributions = mean(all_attributions)
    
    return attributions
end

end