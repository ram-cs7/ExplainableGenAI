using ExplainableGenAI
using BenchmarkTools
using Statistics

# Load model
model, tokenizer = load_model("gpt2")  # or a small model

# Run QA experiment
println("Running QA experiment...")
qa_results = run_qa_experiment(model, tokenizer; methods=[:grad, :attention_rollout, :deletion], max_samples=10)

# Save results
save_results(qa_results, "qa_results.csv")

# Print summary
println("QA Results Summary:")
for method in unique(qa_results.method)
    method_results = filter(row -> row.method == method, qa_results)
    println("Method: $method")
    println("  Mean AUC: ", mean(method_results.auc))
    println("  Mean Comprehensiveness: ", mean(method_results.comprehensiveness))
    println("  Mean Sufficiency: ", mean(method_results.sufficiency))
    println("  Mean Faithfulness Correlation: ", mean(method_results.faithfulness_correlation))
    println("  Mean Runtime (s): ", mean(method_results.runtime))
    println()
end