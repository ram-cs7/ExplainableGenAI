import platform
import psutil
import torch.utils.benchmark as torch_benchmark
import torch
import time
import numpy as np

def print_hardware_info():
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 28 + "\n")

def benchmark():
    print("=== Hardware Information ===")
    print_hardware_info()

    d_model = 768
    d_dict = 3072
    batch_size = 32
    k = 32

    # Set up device - CPU strictly for apples-to-apples comparison
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print(f"Hardware configuration:")
    print(f"d_model={d_model}, d_SAE={d_dict}, batch_size={batch_size}, k={k}")

    # Synthetic input tensors (random initialization for benchmarking)
    x = torch.randn(batch_size, d_model, device=device)
    W_enc = torch.randn(d_model, d_dict, device=device)
    b_enc = torch.randn(d_dict, device=device)
    W_dec = torch.randn(d_dict, d_model, device=device)
    b_dec = torch.randn(d_model, device=device)
    
    log_threshold = torch.randn(d_dict, device=device, requires_grad=True)
    # True order-4 orthogonal group generator (shift by d_model/4)
    shift = d_model // 4
    T_ortho = torch.zeros(d_model, d_model, device=device)
    for i in range(d_model):
        T_ortho[i, (i + shift) % d_model] = 1.0
    group_order = 4

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
        
    equiv_sae_T_cache = []
    Tk_init = torch.eye(d_model, device=device)
    for _ in range(group_order):
        equiv_sae_T_cache.append(Tk_init)
        Tk_init = Tk_init @ T_ortho

    def equiv_sae(x):
        recon_acc = torch.zeros_like(x)
        for i in range(group_order):
            Tk = equiv_sae_T_cache[i]
            x_t = x @ Tk.T
            pre_acts = (x_t - b_dec) @ W_enc + b_enc
            vals, indices = torch.topk(pre_acts, k, dim=-1)
            sparse_acts = torch.relu(torch.zeros_like(pre_acts).scatter_(-1, indices, vals))
            recon_k = sparse_acts @ W_dec + b_dec
            recon_acc = recon_acc + recon_k @ Tk        # undo the transform
        return recon_acc / group_order

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
    
    N_TRIALS = 30  # Number of independent trials for statistics
    N_WARMUP = 5   # Warmup runs (excluded from statistics)

    print(f"--- PyTorch CPU Benchmark Results ({N_TRIALS} trials, {N_WARMUP} warmup) ---")
    print(f"{'Architecture':<20} | {'Forward(ms) ± Std':>19} | {'Backward(ms) ± Std':>19}")
    print("-" * 64)

    for name, model_fn in models.items():
        fw_times = []
        bw_times = []

        for trial in range(N_WARMUP + N_TRIALS):
            # --- Forward pass timing ---
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            y = model_fn(x)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.perf_counter()
            fw_ms = (t1 - t0) * 1000

            # --- Backward pass timing (measured directly, not by subtraction) ---
            loss = y.sum()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t2 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t3 = time.perf_counter()
            bw_ms = (t3 - t2) * 1000

            # Clear gradients for the next trial
            if x.grad is not None:
                x.grad = None
            if log_threshold.grad is not None:
                log_threshold.grad = None

            # Skip warmup trials
            if trial >= N_WARMUP:
                fw_times.append(fw_ms)
                bw_times.append(bw_ms)

        fw_median = float(np.median(fw_times))
        fw_std = float(np.std(fw_times))
        bw_median = float(np.median(bw_times))
        bw_std = float(np.std(bw_times))

        print(f"{name:<20} | {fw_median:8.2f} ±{fw_std:5.2f} ms | {bw_median:8.2f} ±{bw_std:5.2f} ms")

if __name__ == '__main__':
    benchmark()
