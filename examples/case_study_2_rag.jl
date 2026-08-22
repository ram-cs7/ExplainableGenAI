# ==============================================================================
# ILLUSTRATIVE WALKTHROUGH: Multi-Agent Memory Corruption (RAG Pipeline)
# ==============================================================================
# This script demonstrates the ExplainableGenAI.jl intervention pipeline using
# a structured synthetic trace. The executor functions below return fixed outputs
# to illustrate the DAG and SAE intervention APIs. Values are illustrative;
# see paper Section 3.2 for the methodology discussion.
# ==============================================================================

using ExplainableGenAI
using ExplainableGenAI.AgenticTraceability
using Printf

println("==================================================")
println(" Illustrative Walkthrough 2: Multi-Agent Memory Corruption ")
println("==================================================")

# 1. Setup the deterministic executor for the RAG pipeline
# NOTE: This is a synthetic executor with fixed outputs to demonstrate
# the intervention API. It does not connect to a real LLM.
function rag_agent_executor(node::SemanticNode, state::Dict)
    new_state = copy(state)
    
    if node.module_type == :Planning
        # QueryParser
        new_state["parsed_year"] = "2026"
        new_state["parsed_topic"] = "tax brackets"
        new_state["status"] = "success"
        
    elseif node.module_type == :Memory
        # Retriever — simulates Feature #1092 (stale memory cache) being active/ablated
        if get(new_state, "feature_1092_active", true)
            new_state["retrieved_context"] = "Tax brackets for 2023: 10%, 12%, 22%, 24%..."
        else
            new_state["retrieved_context"] = "Tax brackets for 2026: 10%, 12%, 22%, 24% (updated rates)..."
        end
        new_state["status"] = "success"
        
    elseif node.module_type == :Tool
        # Summarizer
        context = get(new_state, "retrieved_context", "")
        if occursin("2023", context)
            new_state["summary"] = "Based on the retrieved context, the 2023 tax brackets are..."
            new_state["prob_2026"] = 0.04
            new_state["kl_divergence"] = 0.0 # Baseline
        else
            new_state["summary"] = "Based on the retrieved context, the 2026 tax brackets are..."
            new_state["prob_2026"] = 0.92
            new_state["kl_divergence"] = 3.85
        end
        
        # Hard override for the oracle DAG intervention
        if get(new_state, "oracle_intervention", false)
            new_state["summary"] = "Based on the oracle context, the 2026 tax brackets are..."
            new_state["prob_2026"] = 0.96
            new_state["kl_divergence"] = 4.21
        end
        
        new_state["status"] = "success"
    end
    
    return new_state
end

# 2. Run the baseline trajectory (Feature #1092 active → failure)
println("\n[1] Running Baseline Trajectory (Failure State)...")
initial_state = Dict("query" => "What are the 2026 tax brackets?", "feature_1092_active" => true)

n_parser = SemanticNode("QueryParser", :Planning, initial_state, "parse", Dict(), 120.0)
state_post_parser = rag_agent_executor(n_parser, initial_state)
n_parser = SemanticNode(n_parser.id, n_parser.module_type, initial_state, "parse", state_post_parser, n_parser.latency_ms)

n_retriever = SemanticNode("Retriever", :Memory, state_post_parser, "retrieve", Dict(), 350.0)
state_post_retriever = rag_agent_executor(n_retriever, state_post_parser)
n_retriever = SemanticNode(n_retriever.id, n_retriever.module_type, state_post_parser, "retrieve", state_post_retriever, n_retriever.latency_ms)

n_summarizer = SemanticNode("Summarizer", :Tool, state_post_retriever, "summarize", Dict(), 600.0)
state_final = rag_agent_executor(n_summarizer, state_post_retriever)
n_summarizer = SemanticNode(n_summarizer.id, n_summarizer.module_type, state_post_retriever, "summarize", state_final, n_summarizer.latency_ms)

trace = AgentTrace("session_rag_2", [n_parser, n_retriever, n_summarizer], :Failure)

println("Baseline P(\"2026\"): $(state_final["prob_2026"])")

# 3. Apply DAG Oracle Intervention: do(v_retriever = z_oracle) (Section 3.2)
println("\n[2] Applying DAG Trace (Oracle Intervention at Retriever)...")
oracle_state = copy(state_post_retriever)
oracle_state["oracle_intervention"] = true # Simulates injecting z_oracle

dag_final_state = rag_agent_executor(n_summarizer, oracle_state)
println("DAG Trace P(\"2026\"): $(dag_final_state["prob_2026"]) | KL-Divergence: $(dag_final_state["kl_divergence"])")

# 4. Apply SAE Feature Ablation (Feature #1092)
println("\n[3] Applying SAE Trace (Ablating Feature #1092 at Retriever)...")
ablated_retriever_state = copy(state_post_parser)
ablated_retriever_state["feature_1092_active"] = false # Simulates ablating the feature

# Re-run Retriever
sae_post_retriever = rag_agent_executor(n_retriever, ablated_retriever_state)
# Re-run Summarizer
sae_final_state = rag_agent_executor(n_summarizer, sae_post_retriever)

println("SAE Trace P(\"2026\"): $(sae_final_state["prob_2026"]) | KL-Divergence: $(sae_final_state["kl_divergence"])")

println("\nWalkthrough complete. Values are illustrative; see paper Section 3.2.")
