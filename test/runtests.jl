using Test
using ExplainableGenAI
using Flux
using Zygote
using Statistics
using LinearAlgebra

# Constrain OpenBLAS to a single thread to prevent out-of-memory (OOM)
# errors in memory-constrained environments (e.g., paging file limits).
BLAS.set_num_threads(1)

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

    @testset "JumpReLU Adjoint Numerical Correctness" begin
        # Finite-difference gradient check for the custom JumpReLU rrule (Lemma 1).
        # This verifies the adjoint is numerically correct, not just non-nil.
        d_model = 16
        d_dict = 64
        sae = JumpReLUSparseAutoencoder(d_model, d_dict; initial_threshold=0.1f0)
        x = randn(Float32, d_model, 4)
        
        # Compute analytic gradient via Zygote (uses our custom rrule)
        loss_fn(m) = sum(m(x)[1])
        _, grads = Flux.withgradient(loss_fn, sae)
        analytic_grad_enc = grads[1][:W_enc]
        
        # Compute finite-difference gradient for a subset of W_enc entries
        eps = 1e-3f0
        for idx in [(1,1), (5,3), (d_dict, d_model)]
            i, j = idx
            sae_plus = deepcopy(sae)
            sae_plus.W_enc[i, j] += eps
            loss_plus = loss_fn(sae_plus)
            
            sae_minus = deepcopy(sae)
            sae_minus.W_enc[i, j] -= eps
            loss_minus = loss_fn(sae_minus)
            
            fd_grad = (loss_plus - loss_minus) / (2 * eps)
            @test isapprox(analytic_grad_enc[i, j], fd_grad; atol=0.1f0)
        end
    end
    
    # Note: Integration tests loading real HuggingFace models via Transformers.jl
    # are excluded from the automated test suite to prevent out-of-memory (OOM) 
    # errors in constrained build environments. The examples/ directory contains
    # illustrative walkthroughs demonstrating the intervention pipeline with
    # structured synthetic traces.
end