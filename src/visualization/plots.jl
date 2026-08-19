module VisualizationPlots

using CairoMakie
using DataFrames
using Statistics

export plot_token_heatmap, plot_attention_flow, plot_deletion_curve, plot_method_comparison, plot_gradcam

"""
    plot_token_heatmap(attributions, input_tokens, generated_tokens; title="Token Attribution Heatmap")

Create a heatmap showing token-to-token attribution scores with color intensity.

# Arguments
- `attributions`: 2D matrix of attribution scores
- `input_tokens`: Vector of input token strings
- `generated_tokens`: Vector of generated token strings
- `title`: Plot title

# Returns
- Makie Figure object
"""
function plot_token_heatmap(attributions, input_tokens, generated_tokens; title="Token Attribution Heatmap")
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Input Tokens", ylabel="Generated Tokens")
    
    heatmap!(ax, attributions, colormap=:viridis)
    
    # Set tick labels
    ax.xticks = (1:length(input_tokens), input_tokens)
    ax.yticks = (1:length(generated_tokens), generated_tokens)
    
    # Rotate x-tick labels
    ax.xticklabelrotation = π/4
    
    return fig
end

"""
    plot_attention_flow(attention_weights, input_tokens, generated_tokens; title="Attention Flow")

Create a visualization of attention flow between tokens.
"""
function plot_attention_flow(attention_weights, input_tokens, generated_tokens; title="Attention Flow")
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], title=title, xlabel="Input Tokens", ylabel="Generated Tokens")
    
    # Create a directed graph visualization
    for (i, gen_token) in enumerate(generated_tokens)
        for (j, in_token) in enumerate(input_tokens)
            weight = attention_weights[i, j]
            if weight > 0.1  # Only show significant connections
                lines!(ax, [j, j], [i, i], color=weight, colormap=:viridis, linewidth=2*weight)
            end
        end
    end
    
    # Set tick labels
    ax.xticks = (1:length(input_tokens), input_tokens)
    ax.yticks = (1:length(generated_tokens), generated_tokens)
    
    # Rotate x-tick labels
    ax.xticklabelrotation = π/4
    
    return fig
end

"""
    plot_deletion_curve(curve; title="Deletion Curve")

Create a line plot showing the deletion curve.
"""
function plot_deletion_curve(curve; title="Deletion Curve")
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], title=title, xlabel="Tokens Removed", ylabel="Probability")
    
    lines!(ax, 0:length(curve)-1, curve, color=:blue, linewidth=2)
    scatter!(ax, 0:length(curve)-1, curve, color=:blue, markersize=8)
    
    # Add AUC annotation
    auc = compute_auc(curve)
    text!(ax, 0.5, 0.9, text="AUC = $(round(auc, digits=3))", align=(:center, :center))
    
    return fig
end

"""
    plot_method_comparison(results_df; metrics=[:auc, :comprehensiveness, :sufficiency])

Create a bar chart comparing different attribution methods.
"""
function plot_method_comparison(results_df; metrics=[:auc, :comprehensiveness, :sufficiency])
    methods = unique(results_df.method)
    
    fig = Figure(size=(800, 600))
    
    for (i, metric) in enumerate(metrics)
        ax = Axis(fig[i, 1], title=string(metric), ylabel=string(metric))
        
        # Calculate mean values for each method
        means = [mean(filter(row -> row.method == m, results_df)[!, metric]) for m in methods]
        method_indices = 1:length(methods)
        
        barplot!(ax, method_indices, means, color=:steelblue)
        ax.xticks = (method_indices, methods)
        
        # Rotate x-tick labels
        ax.xticklabelrotation = π/4
    end
    
    return fig
end

"""
    plot_gradcam(gradcam, original_image; title="Grad-CAM Visualization")

Create a visualization overlaying Grad-CAM heatmap on the original image.
"""
function plot_gradcam(gradcam, original_image; title="Grad-CAM Visualization")
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], title=title)
    
    # Display original image
    image!(ax, original_image)
    
    # Overlay Grad-CAM heatmap with transparency
    heatmap!(ax, gradcam, colormap=:hot, alpha=0.5)
    
    return fig
end

"""
    compute_auc(curve)

Compute Area Under Curve for deletion/insertion curves.
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

end