using ExplainableGenAI
using ExplainableGenAI.AgenticTraceability
using Printf

println("==================================================")
println(" Case Study 1: Math Heuristic Override ")
println("==================================================")

# 1. Setup the deterministic LLM DAG executor
function math_agent_executor(node::SemanticNode, state::Dict)
    new_state = copy(state)
    
    if node.module_type == :Planning
        # Simulating the planner node parsing "What is sqrt(1764) / 3?"
        if get(new_state, "override_feature_402_active", true)
            # The LLM hallucinates an incorrect numerical string instead of invoking CalculatorTool
            new_state["action"] = "direct_answer"
            new_state["answer"] = "14.0" # Incorrect hallucination
            new_state["calculator_tool_prob"] = 0.12 # 12% probability
        else
            # Feature #402 has been clamped/ablated. It successfully routes to the Calculator.
            new_state["action"] = "invoke_calculator"
            new_state["calculator_tool_prob"] = 0.89 # 89% probability
        end
    elseif node.module_type == :Tool
        if new_state["action"] == "invoke_calculator"
            new_state["status"] = "success"
            new_state["answer"] = "14.0" # True calculation (sqrt(1764)/3 = 42/3 = 14)
        else
            new_state["error"] = "Agent hallucinated output instead of calculating."
        end
    end
    
    return new_state
end

# 2. Simulate the initial failure (Feature #402 active)
println("\n[1] Running Baseline Trajectory (Feature #402 Active)...")
initial_state = Dict("query" => "What is sqrt(1764) / 3?", "override_feature_402_active" => true)

n_planner = SemanticNode("planner_1", :Planning, initial_state, "parse_query", Dict(), 450.0)
state_post_planner = math_agent_executor(n_planner, initial_state)
n_planner = SemanticNode(n_planner.id, n_planner.module_type, initial_state, "parse_query", state_post_planner, n_planner.latency_ms)

n_tool = SemanticNode("tool_1", :Tool, state_post_planner, "execute", Dict(), 50.0)
state_final = math_agent_executor(n_tool, state_post_planner)
n_tool = SemanticNode(n_tool.id, n_tool.module_type, state_post_planner, "execute", state_final, n_tool.latency_ms)

trace = AgentTrace("session_math_1", [n_planner, n_tool], :Failure)

println("Baseline CalculatorTool Token Probability: $(state_post_planner["calculator_tool_prob"] * 100)%")
println("Final Output Status: $(get(state_final, "error", "Success"))")

# 3. Apply the mechanistic intervention (clamping Feature #402 to 0.0)
println("\n[2] Applying SAE Intervention: Clamping Feature #402 to 0.0...")
# In the real framework, this translates to modifying the hidden state tensor injected into the node.
# Here we simulate this by flipping the semantic flag parsed by our executor.
oracle_state = copy(state_post_planner)
oracle_state["override_feature_402_active"] = false 

# Rerun the node with the ablated feature
ablated_state = math_agent_executor(n_planner, oracle_state)

println("Counterfactual CalculatorTool Token Probability: $(ablated_state["calculator_tool_prob"] * 100)%")
println("Causal link successfully established: Feature #402 directly induced the macro-agentic failure.")
