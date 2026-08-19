using Pkg
Pkg.activate(".")
using MKL
using ExplainableGenAI
using Flux
using Zygote
using BenchmarkTools
using Printf
using InteractiveUtils
using LinearAlgebra
using Statistics

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
shift = div(d_model, 4)
T_matrix = zeros(Float32, d_model, d_model)
for i in 1:d_model
    T_matrix[i, mod1(i + shift, d_model)] = 1.0f0
end
equiv_sae = EquivariantSparseAutoencoder(d_model, d_dict, T_matrix; k=k)

# Dummy input
println("Generating input data...")
x = randn(Float32, d_model, batch_size)

function benchmark_sae(name, model, input)
    println("Benchmarking $name (Forward Pass)...")
    fw_bench = @benchmark $model($input) samples=10 evals=1
    fw_times_ms = fw_bench.times ./ 1e6
    fw_time_ms = median(fw_times_ms)
    fw_std_ms = std(fw_times_ms)
    
    println("Benchmarking $name (Backward Pass)...")
    bw_bench = @benchmark Zygote.gradient(m -> sum(m($input)[1]), $model) samples=10 evals=1
    bw_times_ms = bw_bench.times ./ 1e6
    bw_time_ms = median(bw_times_ms)
    bw_std_ms = std(bw_times_ms)
    
    @printf("%-20s | %8.2f ±%5.2f ms | %8.2f ±%5.2f ms\n", name, fw_time_ms, fw_std_ms, bw_time_ms, bw_std_ms)
    return fw_time_ms, bw_time_ms
end

target = length(ARGS) > 0 ? ARGS[1] : "all"

println("\n--- Benchmark Results ---")
println(@sprintf("%-20s | %19s | %19s", "Architecture", "Forward(ms) ± Std", "Backward(ms) ± Std"))
println("-"^64)

if target == "topk" || target == "all"
    benchmark_sae("TopK SAE", topk_sae, x)
end
if target == "batchtopk" || target == "all"
    benchmark_sae("BatchTopK SAE", batch_topk_sae, x)
end
if target == "jumprelu" || target == "all"
    benchmark_sae("JumpReLU SAE", jumprelu_sae, x)
end
if target == "equivariant" || target == "all"
    benchmark_sae("Equivariant SAE", equiv_sae, x)
end

println("-"^48)
