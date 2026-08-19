module LLMWrapper

using Transformers
using Transformers.HuggingFace
using Flux

export load_model, generate_with_attention, generate_text

"""
    load_model(model_name::String)

Load a pre-trained model and tokenizer from HuggingFace.

# Arguments
- `model_name`: Name of the HuggingFace model (e.g., "gpt2", "facebook/bart-base")

# Returns
- Tuple of (model, tokenizer)
"""
function load_model(model_name::String)
    model = HuggingFace.load_model(model_name)
    tokenizer = HuggingFace.load_tokenizer(model_name)
    return model, tokenizer
end

"""
    generate_with_attention(model, tokenizer, input_text::String; max_length=20)

Generate text and return the output along with attention weights.
"""
function generate_with_attention(model, tokenizer, input_text::String; max_length=20)
    # Tokenize input
    input_tokens = tokenizer(input_text)
    input_ids = input_tokens.token
    
    # Initialize storage
    generated_ids = copy(input_ids)
    attentions = []
    
    # Generation loop
    for i in 1:max_length
        # Get model output with attention
        output = model(generated_ids; output_attentions=true)
        logits = output.logits
        last_token_logits = logits[:, end, :]
        next_token = argmax(last_token_logits)
        
        # Store attention weights for this step
        push!(attentions, output.attentions)
        
        # Append next token
        push!(generated_ids, next_token)
    end
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text, attentions, generated_ids
end

"""
    generate_text(model, tokenizer, input_text::String; max_length=20, temperature=1.0, top_k=50, top_p=0.9)

Generate text with advanced sampling strategies (temperature, top-k, nucleus sampling).
This is a more flexible generation function compared to generate_with_attention.

# Arguments
- `model`: The language model
- `tokenizer`: The tokenizer
- `input_text`: Input prompt string
- `max_length`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (higher = more random)
- `top_k`: Keep only top k tokens for sampling
- `top_p`: Nucleus sampling threshold (keep tokens with cumulative probability >= top_p)

# Returns
- Generated text string
"""
function generate_text(model, tokenizer, input_text::String; max_length=20, temperature=1.0, top_k=50, top_p=0.9)
    # Tokenize input
    input_tokens = tokenizer(input_text)
    input_ids = input_tokens.token
    
    # Initialize generated sequence
    generated_ids = copy(input_ids)
    
    # Generation loop
    for i in 1:max_length
        # Get model output
        logits = model(generated_ids)
        last_token_logits = logits[:, end, :]
        
        # Apply temperature
        scaled_logits = last_token_logits ./ temperature
        
        # Convert to probabilities
        probs = Flux.softmax(scaled_logits)
        
        # Top-k filtering
        if top_k > 0
            top_k_indices = partialsortperm(vec(probs), 1:min(top_k, length(probs)), rev=true)
            filtered_probs = zeros(Float32, size(probs))
            filtered_probs[top_k_indices] = probs[top_k_indices]
            probs = filtered_probs ./ sum(filtered_probs)
        end
        
        # Nucleus (top-p) sampling
        if top_p < 1.0
            sorted_indices = sortperm(vec(probs), rev=true)
            cumsum_probs = cumsum(probs[sorted_indices])
            nucleus_size = findfirst(x -> x >= top_p, cumsum_probs)
            if nucleus_size !== nothing
                nucleus_indices = sorted_indices[1:nucleus_size]
                filtered_probs = zeros(Float32, size(probs))
                filtered_probs[nucleus_indices] = probs[nucleus_indices]
                probs = filtered_probs ./ sum(filtered_probs)
            end
        end
        
        # Sample next token
        next_token = sample_token(probs)
        
        # Append to sequence
        push!(generated_ids, next_token)
        
        # Stop if EOS token (assuming token id 50256 for GPT-2, adjust as needed)
        if next_token == 50256
            break
        end
    end
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text
end

"""
    sample_token(probs)

Sample a token index from a probability distribution.
"""
function sample_token(probs::AbstractArray)
    # Multinomial sampling
    cumsum_probs = cumsum(vec(probs))
    r = rand()
    idx = findfirst(x -> x >= r, cumsum_probs)
    return idx === nothing ? length(probs) : idx
end

end