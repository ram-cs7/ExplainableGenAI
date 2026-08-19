using Test
using ExplainableGenAI
using Flux
using Zygote
using Statistics

@testset "ExplainableGenAI.jl 2026 SOTA Core Tests" begin

    @testset "Mechanistic Interpretability (SAE)" begin
        # Test SAE Initialization
        d_model = 64
        d_dict = 256
        k = 8
        sae = SparseAutoencoder(d_model, d_dict, k=k)
        
        @test size(sae.W_enc) == (d_dict, d_model)
        @test size(sae.W_dec) == (d_model, d_dict)
        
        # Test Forward Pass & TopK Spacing
        batch_size = 4
        x = randn(Float32, d_model, batch_size)
        reconstructed, acts = sae(x)
        
        @test size(reconstructed) == (d_model, batch_size)
        @test size(acts) == (d_dict, batch_size)
        
        # Verify strict TopK sparsity
        for b in 1:batch_size
            non_zeros = count(acts[:, b] .> 0)
            @test non_zeros <= k
        end
        
        # Test Zygote Differentiability (Crucial for Julia SAE training)
        loss_val, grads = Flux.withgradient(sae) do m
            reconstructed, acts = m(x)
            mean((reconstructed .- x).^2)
        end
        @test grads[1][:W_enc] !== nothing
        @test grads[1][:W_dec] !== nothing
    end

    @testset "Agentic Traceability" begin
        # Construct an AgentTrace
        n1 = SemanticNode("1", :Memory, Dict(), "search_db", Dict("docs"=>[]), 10.0)
        n2 = SemanticNode("2", :Planning, Dict(), "plan", Dict("error"=>"timeout"), 20.0)
        trace = AgentTrace("session_01", [n1, n2], :Failure)
        
        # Mock executor for true counterfactual replay
        executor = (node, state) -> begin
            if node.id == "2" && haskey(state, "status") && state["status"] == "success"
                return state # The injected successful state propagates
            end
            return node.output_state
        end
        
        # Test True Counterfactual Replay
        scores = diagnose_failure(trace, agent_executor=executor)
        @test scores[:Planning] == 1.0
        @test scores[:Memory] == 0.0
    end
    
    # Note on LLMWrapper/HookedTransformer:
    # Integration tests loading real HuggingFace models via Transformers.jl
    # are excluded from the automated CI suite to prevent out-of-memory (OOM) 
    # errors in constrained build environments. See examples/2026_sota_demo.jl 
    # for executable end-to-end paths on supported hardware.
end