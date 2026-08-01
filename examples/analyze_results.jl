using ExplainableGenAI
using DataFrames
using CSV
using CairoMakie
using Statistics

# Load results
results = CSV.read("qa_results.csv", DataFrame)

# Create comparison plots
fig = Figure(size=(1200, 800))

# Get unique methods
methods = unique(results.method)
method_indices = 1:length(methods)

# AUC comparison
ax1 = Axis(fig[1, 1], title="AUC Comparison", ylabel="AUC", xlabel="Method")
auc_means = [mean(filter(row -> row.method == m, results).auc) for m in methods]
barplot!(ax1, method_indices, auc_means, color=:steelblue)
ax1.xticks = (method_indices, methods)
ax1.xticklabelrotation = π/4

# Comprehensiveness comparison
ax2 = Axis(fig[1, 2], title="Comprehensiveness Comparison", ylabel="Comprehensiveness", xlabel="Method")
comp_means = [mean(filter(row -> row.method == m, results).comprehensiveness) for m in methods]
barplot!(ax2, method_indices, comp_means, color=:coral)
ax2.xticks = (method_indices, methods)
ax2.xticklabelrotation = π/4

# Sufficiency comparison
ax3 = Axis(fig[2, 1], title="Sufficiency Comparison", ylabel="Sufficiency", xlabel="Method")
suff_means = [mean(filter(row -> row.method == m, results).sufficiency) for m in methods]
barplot!(ax3, method_indices, suff_means, color=:seagreen)
ax3.xticks = (method_indices, methods)
ax3.xticklabelrotation = π/4

# Faithfulness correlation comparison
ax4 = Axis(fig[2, 2], title="Faithfulness Correlation Comparison", ylabel="Correlation", xlabel="Method")
corr_means = [mean(filter(row -> row.method == m, results).faithfulness_correlation) for m in methods]
barplot!(ax4, method_indices, corr_means, color=:orchid)
ax4.xticks = (method_indices, methods)
ax4.xticklabelrotation = π/4

# Runtime comparison
ax5 = Axis(fig[3, 1:2], title="Runtime Comparison", ylabel="Runtime (s)", xlabel="Method")
runtime_means = [mean(filter(row -> row.method == m, results).runtime) for m in methods]
barplot!(ax5, method_indices, runtime_means, color=:crimson)
ax5.xticks = (method_indices, methods)
ax5.xticklabelrotation = π/4

# Save figure
save("method_comparison.png", fig)

# Print summary statistics
println("Summary Statistics:")
for method in methods
    method_results = filter(row -> row.method == method, results)
    println("\nMethod: $method")
    println("  AUC: $(mean(method_results.auc)) ± $(std(method_results.auc))")
    println("  Comprehensiveness: $(mean(method_results.comprehensiveness)) ± $(std(method_results.comprehensiveness))")
    println("  Sufficiency: $(mean(method_results.sufficiency)) ± $(std(method_results.sufficiency))")
    println("  Faithfulness Correlation: $(mean(method_results.faithfulness_correlation)) ± $(std(method_results.faithfulness_correlation))")
    println("  Runtime: $(mean(method_results.runtime)) ± $(std(method_results.runtime)) seconds")
end