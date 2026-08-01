using ExplainableGenAI
using CairoMakie

# Load model
model, tokenizer = load_model("gpt2")  # or a small model

# Input text
input_text = "The quick brown fox jumps over the lazy dog"

# Generate text and get attention
generated_text, attentions, generated_ids = generate_with_attention(model, tokenizer, input_text; max_length=10)

println("Input: ", input_text)
println("Generated: ", generated_text)

# Get input token IDs
input_tokens = tokenizer(input_text)
input_ids = input_tokens.token

# Explain first generated token (position = length(input_ids) + 1)
target_position = length(input_ids) + 1

# Compute gradient attribution
grad_attributions = compute_gradient_attributions(model, tokenizer, input_ids, generated_ids, target_position)

# Compute integrated gradients
ig_attributions = compute_integrated_gradients(model, tokenizer, input_ids, generated_ids, target_position)

# Visualize attributions
token_texts = [tokenizer.decode([id]) for id in input_ids]

# Create figure with bar plots
fig = Figure(size=(1000, 500))

# Gradient * Input plot
ax1 = Axis(fig[1, 1], 
    title = "Gradient × Input Attribution",
    xlabel = "Input Token",
    ylabel = "Attribution Score")
barplot!(ax1, 1:length(input_ids), grad_attributions, color=:steelblue)
ax1.xticks = (1:length(input_ids), token_texts)
ax1.xticklabelrotation = π/4

# Integrated Gradients plot
ax2 = Axis(fig[1, 2],
    title = "Integrated Gradients Attribution",
    xlabel = "Input Token",
    ylabel = "Attribution Score")
barplot!(ax2, 1:length(input_ids), ig_attributions, color=:coral)
ax2.xticks = (1:length(input_ids), token_texts)
ax2.xticklabelrotation = π/4

# Save and display
save("attribution_comparison.png", fig)
display(fig)