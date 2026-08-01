module Dashboard

using PlutoUI
using CairoMakie
using DataFrames
using CSV
using Markdown

export create_interactive_dashboard

"""
    create_interactive_dashboard()

Create an interactive Pluto notebook dashboard for exploring token attributions.
This function returns markdown content that can be used in a Pluto.jl notebook.

# Returns
- Markdown string with interactive dashboard components
"""
function create_interactive_dashboard()
    return md"""
# Explainable GenAI Interactive Dashboard

This dashboard allows you to explore token attributions for generative models.

## Model Selection
```julia
@bind model_name Select(["gpt2", "facebook/bart-base", "t5-small"])
```

## Input Text
```julia
@bind input_text TextField(default="The future of AI is")
```

## Generation Parameters
```julia
@bind max_length Slider(5:50, default=20)
@bind temperature Slider(0.1:0.1:2.0, default=1.0)
```

## Attribution Method
```julia
@bind method Select(["grad", "ig", "smoothgrad", "attention_rollout", "attention_flow", "deletion", "insertion"])
```

## Generate and Explain
```julia
begin
    # Load model (cached)
    if !@isdefined(model) || model_name != last_model_name
        model, tokenizer = load_model(model_name)
        last_model_name = model_name
    end
    
    # Generate text with attention
    generated_text, attentions, generated_ids = generate_with_attention(model, tokenizer, input_text; max_length=max_length)
    
    # Compute attributions
    attributions = explain(model, tokenizer, input_text, generated_text; method=Symbol(method))
    
    # Get input tokens
    input_tokens = tokenizer(input_text)
    input_ids = input_tokens.token
    
    # Get first generated token for visualization
    target_pos = length(input_ids) + 1
    attr_dict = attributions[target_pos]
    attr_scores = attr_dict[:attributions]
    token_texts = attr_dict[:input_tokens]
    
    # Compute deletion curve
    del_curve = compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_pos)
    auc = compute_auc(del_curve)
    
    # Display results
    HTML("
    <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
        <h3>Generation Results</h3>
        <p><strong>Input:</strong> $(input_text)</p>
        <p><strong>Generated:</strong> $(generated_text)</p>
    </div>
    ")
end
```

## Token Attribution Visualization
```julia
begin
    # Create bar plot for token attributions
    fig1 = Figure(size=(800, 400))
    ax1 = Axis(fig1[1, 1], title="Token Attribution for First Generated Token", xlabel="Input Token", ylabel="Attribution Score")
    barplot!(ax1, 1:length(attr_scores), attr_scores)
    ax1.xticks = (1:length(token_texts), token_texts)
    ax1.xticklabelrotation = π/4
    fig1
end
```

## Deletion Curve
```julia
begin
    # Create deletion curve plot
    fig2 = Figure(size=(800, 400))
    ax2 = Axis(fig2[1, 1], title="Deletion Curve (AUC = $auc)", xlabel="Tokens Removed", ylabel="Probability")
    lines!(ax2, 0:length(del_curve)-1, del_curve)
    scatter!(ax2, 0:length(del_curve)-1, del_curve)
    fig2
end
```

## Attention Visualization (if applicable)
```julia
if method in ["attention_rollout", "attention_flow"]
    begin
        # Get attention weights
        if method == "attention_rollout"
            attention_weights = compute_attention_rollout(attentions)
        else
            attention_weights = compute_attention_flow(attentions)
        end
        
        # Get generated tokens
        generated_tokens = tokenizer.decode(generated_ids[length(input_ids)+1:end])
        
        # Create attention heatmap
        fig3 = plot_token_heatmap(
            attention_weights[length(input_ids)+1:end, 1:length(input_ids)], 
            token_texts, 
            generated_tokens;
            title="Attention Heatmap"
        )
        fig3
    end
end
```

## Evaluation Metrics
```julia
begin
    # Compute metrics
    comp = compute_comprehensiveness(model, tokenizer, input_ids, generated_ids, target_pos, attr_scores)
    suff = compute_sufficiency(model, tokenizer, input_ids, generated_ids, target_pos, attr_scores)
    
    # Display metrics
    HTML("
    <div style='padding: 10px; background-color: #e6f7ff; border-radius: 5px;'>
        <h3>Evaluation Metrics</h3>
        <p><strong>Comprehensiveness:</strong> $(round(comp, digits=4))</p>
        <p><strong>Sufficiency:</strong> $(round(suff, digits=4))</p>
        <p><strong>Deletion AUC:</strong> $(round(auc, digits=4))</p>
    </div>
    ")
end
```

## Export Results
```julia
begin
    # Create export button
    @bind export_button Button("Export Results")
    
    if export_button
        # Save results to CSV
        results_df = DataFrame(
            input_token=token_texts,
            attribution_score=attr_scores
        )
        CSV.write("attribution_results.csv", results_df)
        
        # Save plots
        save("token_attributions.png", fig1)
        save("deletion_curve.png", fig2)
        
        HTML("
        <div style='padding: 10px; background-color: #f6ffed; border-radius: 5px;'>
            <h3>Export Complete</h3>
            <p>Results saved to:</p>
            <ul>
                <li>attribution_results.csv</li>
                <li>token_attributions.png</li>
                <li>deletion_curve.png</li>
            </ul>
        </div>
        ")
    end
end
```
"""
end

end
   