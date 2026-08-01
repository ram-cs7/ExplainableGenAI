import torch
import time

def benchmark():
    d_model = 768
    d_dict = 3072
    batch_size = 32
    k = 32

    # Set up device - CPU strictly for apples-to-apples comparison
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print(f"Hardware configuration:")
    print(f"d_model={d_model}, d_SAE={d_dict}, batch_size={batch_size}, k={k}")

    # Dummy data
    x = torch.randn(batch_size, d_model, device=device)
    W_enc = torch.randn(d_model, d_dict, device=device)
    b_enc = torch.randn(d_dict, device=device)
    W_dec = torch.randn(d_dict, d_model, device=device)
    b_dec = torch.randn(d_model, device=device)
    
    log_threshold = torch.randn(d_dict, device=device, requires_grad=True)
    T_matrix = torch.randn(d_model, d_model, device=device)

    def topk_sae(x):
        pre_acts = (x - b_dec) @ W_enc + b_enc
        vals, indices = torch.topk(pre_acts, k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts).scatter_(-1, indices, vals)
        sparse_acts = torch.relu(sparse_acts)
        return sparse_acts @ W_dec + b_dec
        
    def batch_topk_sae(x):
        pre_acts = (x - b_dec) @ W_enc + b_enc
        total_budget = k * x.shape[0]
        # In CPU, topk on flattened array
        vals, _ = torch.topk(pre_acts.flatten(), total_budget)
        threshold = vals[-1]
        sparse_acts = torch.where(pre_acts >= threshold, pre_acts, torch.zeros_like(pre_acts))
        sparse_acts = torch.relu(sparse_acts)
        return sparse_acts @ W_dec + b_dec
        
    def jumprelu_sae(x):
        pre_acts = (x - b_dec) @ W_enc + b_enc
        threshold = torch.exp(log_threshold)
        sparse_acts = torch.where(pre_acts > threshold, pre_acts, torch.zeros_like(pre_acts))
        return sparse_acts @ W_dec + b_dec
        
    def equiv_sae(x):
        pre_acts = (x - b_dec) @ W_enc + b_enc
        vals, indices = torch.topk(pre_acts, k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts).scatter_(-1, indices, vals)
        sparse_acts = torch.relu(sparse_acts)
        return sparse_acts @ W_dec + b_dec

    x.requires_grad_(True)
    W_enc.requires_grad_(True)
    b_enc.requires_grad_(True)
    W_dec.requires_grad_(True)
    b_dec.requires_grad_(True)

    models = {
        "TopK SAE": topk_sae,
        "BatchTopK SAE": batch_topk_sae,
        "JumpReLU SAE": jumprelu_sae,
        "Equivariant SAE": equiv_sae
    }
    
    print("\n--- PyTorch CPU Benchmark Results ---")
    print(f"{'Architecture':<20} | {'Forward(ms)':>11} | {'Backward(ms)':>11}")
    print("-" * 48)

    for name, model_fn in models.items():
        # Warmup
        for _ in range(5):
            y = model_fn(x)
            loss = y.sum()
            loss.backward()

        # Benchmark forward
        t0 = time.perf_counter()
        for _ in range(100):
            y = model_fn(x)
        fw_time = (time.perf_counter() - t0) * 1000 / 100

        # Benchmark backward
        t0 = time.perf_counter()
        for _ in range(100):
            y = model_fn(x)
            loss = y.sum()
            loss.backward()
        bw_time = (time.perf_counter() - t0) * 1000 / 100 - fw_time

        print(f"{name:<20} | {fw_time:8.2f} ms | {bw_time:8.2f} ms")

if __name__ == '__main__':
    benchmark()
