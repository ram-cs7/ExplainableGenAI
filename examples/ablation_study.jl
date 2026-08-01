using ExplainableGenAI
using DataFrames
using CSV
using Statistics

# Load model
model, tokenizer = load_model("gpt2")

# Ablation 1: Integrated gradients with different steps
println("Ablation 1: Integrated gradients with different steps")
steps_list = [5, 10, 20, 50]
ig_results = DataFrame(steps=Int[], auc=Float32[], runtime=Float32[])

# Use a single example for speed
example = load_dataset("squad", max_samples=1)[1]
input_text = example["question"] * " [SEP] " * example["context"]
generated_text, _, generated_ids = generate_with_attention(model, tokenizer, input_text; max_length=20)
input_tokens = tokenizer(input_text)
input_ids = input_tokens.token
target_pos = length(input_ids) + 1

for steps in steps_list
    println("  Testing with $steps steps...")
    
    # Time the attribution computation
    runtime = @elapsed attributions = compute_integrated_gradients(model, tokenizer, input_ids, generated_ids, target_pos; steps=steps)
    
    # Compute deletion curve
    del_curve = compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_pos)
    auc = compute_auc(del_curve)
    
    push!(ig_results, Dict(:steps => steps, :auc => auc, :runtime => runtime))
end

# Save results
CSV.write("ig_ablation.csv", ig_results)

# Ablation 2: Attention rollout with different discard ratios
println("\nAblation 2: Attention rollout with different discard ratios")
discard_ratios = [0.0, 0.1, 0.2, 0.3]
attn_results = DataFrame(discard_ratio=Float32[], auc=Float32[], runtime=Float32[])

# Generate with attention
_, attentions, _ = generate_with_attention(model, tokenizer, input_text; max_length=20)

for discard_ratio in discard_ratios
    println("  Testing with discard ratio $discard_ratio...")
    
    # Time the attribution computation
    runtime = @elapsed rollout = compute_attention_rollout(attentions; discard_ratio=discard_ratio)
    
    # Get attributions
    attr_scores = rollout[target_pos, 1:length(input_ids)]
    
    # Compute deletion curve
    del_curve = compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_pos)
    auc = compute_auc(del_curve)
    
    push!(attn_results, Dict(:discard_ratio => discard_ratio, :auc => auc, :runtime => runtime))
end

# Save results
CSV.write("attention_ablation.csv", attn_results)

# Print summaries
println("\nIntegrated Gradients Ablation:")
println(ig_results)

println("\nAttention Rollout Ablation:")
println(attn_results)