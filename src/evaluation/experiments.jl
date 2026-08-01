module EvaluationExperiments

using ..ExplainableGenAI
using ..EvaluationMetrics
using DataFrames
using CSV
using JSON
using HTTP
using BenchmarkTools

export run_qa_experiment, run_summarization_experiment, save_results, load_dataset

"""
    load_dataset(dataset_name; split="validation", max_samples=100)

Load a dataset for evaluation.
"""
function load_dataset(dataset_name; split="validation", max_samples=100)
    if dataset_name == "squad"
        # Load SQuAD dataset
        url = "https://raw.githubusercontent.com/stanfordnlp/squad/master/data/v1.1/$(split)-v1.1.json"
        response = HTTP.get(url)
        data = JSON.parse(String(response.body))
        
        # Extract examples
        examples = []
        for paragraph in data["data"]
            for qa in paragraph["paragraphs"][1]["qas"]
                if length(examples) >= max_samples
                    return examples
                end
                
                question = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                context = paragraph["paragraphs"][1]["context"]
                
                push!(examples, Dict(
                    :question => question,
                    :answers => answers,
                    :context => context
                ))
            end
        end
        
        return examples
    else
        throw(ArgumentError("Dataset $dataset_name is not supported or requires HuggingFace datasets integration without mocking."))
    end
end

"""
    run_qa_experiment(model, tokenizer; methods=[:grad, :attention_rollout, :deletion], max_samples=20)

Run experiments on SQuAD dataset.
"""
function run_qa_experiment(model, tokenizer; methods=[:grad, :attention_rollout, :deletion], max_samples=20)
    # Load dataset
    examples = load_dataset("squad", max_samples=max_samples)
    
    results = DataFrame(
        example_id=Int[],
        method=String[],
        target_position=Int[],
        auc=Float32[],
        comprehensiveness=Float32[],
        sufficiency=Float32[],
        faithfulness_correlation=Float32[],
        runtime=Float32[]
    )
    
    for (i, example) in enumerate(examples)
        println("Processing example $i/$(length(examples))")
        
        # Combine question and context
        input_text = example["question"] * " [SEP] " * example["context"]
        
        # Generate answer (simplified - in practice would use a proper QA model)
        generated_text, _, generated_ids = generate_with_attention(model, tokenizer, input_text; max_length=50)
        
        # Get input tokens
        input_tokens = tokenizer(input_text)
        input_ids = input_tokens.token
        
        # Find answer tokens in generated text (simplified)
        answer = example["answers"][1]
        answer_tokens = tokenizer(answer)
        answer_ids = answer_tokens.token
        
        # Find position of answer in generated text (simplified)
        target_positions = []
        for pos in (length(input_ids)+1):length(generated_ids)
            if generated_ids[pos] == answer_ids[1]  # Simplified - just match first token
                push!(target_positions, pos)
            end
        end
        
        if isempty(target_positions)
            println("Answer not found in generated text, skipping")
            continue
        end
        
        # Evaluate each method
        for method in methods
            println("  Evaluating method: $method")
            
            # Time the attribution computation
            runtime = @elapsed attributions = explain(model, tokenizer, input_text, generated_text; method=method, target_tokens=[1])
            
            # Get attributions for first target position
            attr_dict = attributions[target_positions[1]]
            attr_scores = attr_dict[:attributions]
            
            # Compute metrics
            # Deletion curve
            del_curve = compute_deletion_curve(model, tokenizer, input_ids, generated_ids, target_positions[1])
            auc = compute_auc(del_curve)
            
            # Comprehensiveness
            comp = compute_comprehensiveness(model, tokenizer, input_ids, generated_ids, target_positions[1], attr_scores)
            
            # Sufficiency
            suff = compute_sufficiency(model, tokenizer, input_ids, generated_ids, target_positions[1], attr_scores)
            
            # Faithfulness correlation
            corr = compute_faithfulness_correlation(model, tokenizer, input_ids, generated_ids, target_positions[1], attr_scores)
            
            # Store results
            push!(results, Dict(
                :example_id => i,
                :method => string(method),
                :target_position => target_positions[1],
                :auc => auc,
                :comprehensiveness => comp,
                :sufficiency => suff,
                :faithfulness_correlation => corr,
                :runtime => runtime
            ))
        end
    end
    
    return results
end

"""
    save_results(results, filename)

Save results to CSV file.
"""
function save_results(results, filename)
    CSV.write(filename, results)
end

end