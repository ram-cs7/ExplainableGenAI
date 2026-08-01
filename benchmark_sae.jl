using Pkg
Pkg.activate(".")
using ExplainableGenAI
using Flux
using Zygote
using BenchmarkTools
using Printf
using InteractiveUtils

# Simulate typical SAE dimensions
const d_model = 768
const d_dict = 3072
const batch_size = 32
const k = 32

println("=== Hardware Information ===")
versioninfo()
println("============================\n")

# Initialize models
println("Initializing SAEs...")
topk_sae = SparseAutoencoder(d_model, d_dict; k=k)
batch_topk_sae = BatchTopKSparseAutoencoder(d_model, d_dict; k=k)
jumprelu_sae = JumpReLUSparseAutoencoder(d_model, d_dict; initial_threshold=0.1f0)
T_matrix = randn(Float32, d_model, d_model)
equiv_sae = EquivariantSparseAutoencoder(d_model, d_dict, T_matrix; k=k)

# Dummy input
println("Generating input data...")
x = randn(Float32, d_model, batch_size)

function benchmark_sae(name, model, input)
    println("Benchmarking $name (Forward Pass)...")
    fw_bench = @benchmark $model($input) samples=10 evals=1
    fw_time_ms = median(fw_bench).time / 1e6
    
    println("Benchmarking $name (Backward Pass)...")
    bw_bench = @benchmark Zygote.gradient(m -> sum(m($input)[1]), $model) samples=10 evals=1
    bw_time_ms = median(bw_bench).time / 1e6
    
    @printf("%-20s | %8.2f ms | %8.2f ms\n", name, fw_time_ms, bw_time_ms)
    return fw_time_ms, bw_time_ms
end

println("\n--- Benchmark Results ---")
println(@sprintf("%-20s | %11s | %11s", "Architecture", "Forward(ms)", "Backward(ms)"))
println("-"^48)

benchmark_sae("TopK SAE", topk_sae, x)
benchmark_sae("BatchTopK SAE", batch_topk_sae, x)
benchmark_sae("JumpReLU SAE", jumprelu_sae, x)
benchmark_sae("Equivariant SAE", equiv_sae, x)

println("-"^48)
