module ExplainableGenAI

# Include all submodules first (before using them)
include("models/llm_wrapper.jl")
include("models/img_wrapper.jl")
include("attribution/attention.jl")
include("attribution/gradient.jl")
include("attribution/perturbation.jl")
include("attribution/mechanistic.jl")
include("attribution/agentic.jl")
include("evaluation/metrics.jl")
include("evaluation/experiments.jl")
include("evaluation/faithfulness.jl")
include("visualization/plots.jl")
include("visualization/dashboard.jl")
include("attribution/autoresearch.jl")

# Now import from submodules
using .LLMWrapper
using .ImageWrapper
using .AttentionAttribution
using .GradientAttribution
using .PerturbationAttribution
using .MechanisticInterpretability
using .AgenticTraceability
using .EvaluationMetrics
using .EvaluationExperiments
using .FaithfulnessEvaluation
using .VisualizationPlots
using .Dashboard
using .Autoresearch

# Export all public functions
export load_model, generate_with_attention, explain, load_image_model
export compute_gradient_attributions, compute_integrated_gradients, compute_smoothgrad
export compute_attention_rollout, compute_attention_flow
export compute_perturbation_attributions, compute_deletion_curve, compute_insertion_curve
export HookedTransformer, cache_activations, patch_activations
export SparseAutoencoder, train_sae!, steer_features, causal_scrub_head
export BatchTopKSparseAutoencoder, EquivariantSparseAutoencoder
export JumpReLUSparseAutoencoder, jumprelu_sae_loss
export SemanticNode, AgentTrace, counterfactual_replay, diagnose_failure
export compute_auc, compute_comprehensiveness, compute_sufficiency, compute_faithfulness_correlation
export compute_intervention_faithfulness, logit_kl_divergence, evaluate_cot_faithfulness
export run_qa_experiment, run_summarization_experiment, save_results, load_dataset
export plot_token_heatmap, plot_attention_flow, plot_deletion_curve, plot_method_comparison, plot_gradcam
export create_interactive_dashboard, generate_image_with_attribution, compute_gradcam
export AutomatedInterpretabilityAgent, auto_discover_features

"""
    explain(model, tokenizer, input_text, generated_text; method=:grad, target_tokens=nothing)

Compute token attributions using specified method.

# Arguments
- `model`: The language model
- `tokenizer`: The tokenizer
- `input_text`: Input text string
- `generated_text`: Generated text string
- `method`: Attribution method (:grad, :ig, :smoothgrad, :attention_rollout, :attention_flow, :deletion, :insertion)
- `target_tokens`: Indices of target tokens to explain (default: all generated tokens)

# Returns
- Dictionary with attribution scores and metadata
"""
function explain(model, tokenizer, input_text, generated_text; method=:grad, target_tokens=nothing)
    # Tokenize input and generated text
    input_tokens = tokenizer(input_text)
    input_ids = input_tokens.token
    
    # For generated text, we need the full sequence (input + generated)
    full_text = input_text * " " * generated_text
    full_tokens = tokenizer(full_text)
    generated_ids = full_tokens.token
    
    # Determine target positions
    if target_tokens === nothing
        # Default to all generated tokens (ensure we have at least one)
        start_pos = length(input_ids) + 1
        end_pos = length(generated_ids)
        if start_pos <= end_pos
            target_positions = collect(start_pos:end_pos)
        else
            # If no new tokens generated, use last position
            target_positions = [length(input_ids)]
        end
    else
        # Convert token indices to positions
        target_positions = [length(input_ids) + i for i in target_tokens]
    end
    
    # Compute attributions for each target position
    results = Dict()
    
    for target_pos in target_positions
        if method == :grad
            attributions = compute_gradient_attributions(model, tokenizer, input_ids, generated_ids, target_pos)
        elseif method == :ig
            attributions = compute_integrated_gradients(model, tokenizer, input_ids, generated_ids, target_pos)
        elseif method == :smoothgrad
            attributions = compute_smoothgrad(model, tokenizer, input_ids, generated_ids, target_pos)
        elseif method == :attention_rollout
            # Need to generate with attention first
            _, attentions, _ = generate_with_attention(model, tokenizer, input_text; max_length=length(generated_ids)-length(input_ids))
            rollout = compute_attention_rollout(attentions)
            # Get attention from target position to input tokens
            attributions = rollout[target_pos, 1:length(input_ids)]
        elseif method == :attention_flow
            # Need to generate with attention first
            _, attentions, _ = generate_with_attention(model, tokenizer, input_text; max_length=length(generated_ids)-length(input_ids))
            flow = compute_attention_flow(attentions)
            # Get attention from target position to input tokens
            attributions = flow[target_pos, 1:length(input_ids)]
        elseif method == :deletion
            attributions = compute_perturbation_attributions(model, tokenizer, input_ids, generated_ids, target_pos; method="deletion")
        elseif method == :insertion
            attributions = compute_perturbation_attributions(model, tokenizer, input_ids, generated_ids, target_pos; method="insertion")
        else
            println("Warning: Unknown method $method. Falling back to gradient baseline.")
            method_func = GradientAttribution.compute_gradient_attributions
        end
        
        # Store results
        results[target_pos] = Dict(
            :attributions => attributions,
            :input_tokens => tokenizer.decode(input_ids),
            :target_token => tokenizer.decode([generated_ids[target_pos]])[1],
            :method => method
        )
    end
    
    return results
end

end