import os
import torch
import numpy as np
import time
import smplx # Assuming this library is available for ground truth
# import cuda_smplx_ops # This will be the compiled CUDA extension

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not found. Plotting will be skipped.")

# Model configuration
MODEL_PATH_HERE = os.environ.get('SMPLX_TEST_MODELS_PATH', 'data/smplx_models')
MODEL_TYPE = 'smplx'
GENDER = 'neutral'
NUM_BETAS = 10
NUM_EXPRESSION_COEFFS = 10
NUM_BODY_JOINTS = 21 
NUM_HAND_JOINTS = 15
NUM_JAW_JOINTS = 1
DTYPE = torch.float32

# Benchmarking configuration
BATCH_SIZES_TO_TEST = [1, 2, 4, 8, 16, 32, 64] # Example batch sizes
NUM_WARMUP_RUNS = 5
NUM_TIMED_RUNS = 20

def get_sample_inputs_bench(batch_size, device='cuda'):
    # Using float32 for benchmarking actual performance
    betas = torch.randn(batch_size, NUM_BETAS, dtype=DTYPE, device=device)
    expression = torch.randn(batch_size, NUM_EXPRESSION_COEFFS, dtype=DTYPE, device=device)
    global_orient = torch.randn(batch_size, 3, dtype=DTYPE, device=device) * 0.1
    body_pose = torch.randn(batch_size, NUM_BODY_JOINTS * 3, dtype=DTYPE, device=device) * 0.1
    jaw_pose = torch.randn(batch_size, NUM_JAW_JOINTS * 3, dtype=DTYPE, device=device) * 0.1
    left_hand_pose = torch.randn(batch_size, NUM_HAND_JOINTS * 3, dtype=DTYPE, device=device) * 0.1
    right_hand_pose = torch.randn(batch_size, NUM_HAND_JOINTS * 3, dtype=DTYPE, device=device) * 0.1
    return betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose

def benchmark_smplx_versions():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return
    device = torch.device("cuda:0")

    try:
        import cuda_smplx_ops
    except ImportError:
        print("CUDA extension 'cuda_smplx_ops' not found. Compile first. Skipping benchmark.")
        return

    print(f"Benchmarking on device: {device}")
    print(f"Warmup runs: {NUM_WARMUP_RUNS}, Timed runs: {NUM_TIMED_RUNS}")

    # 1. Load reference SMPL-X model and prepare CUDA constants
    try:
        ref_model_params_src = smplx.create(
            MODEL_PATH_HERE, model_type=MODEL_TYPE, gender=GENDER, use_pca=False,
            num_betas=NUM_BETAS, num_expression_coeffs=NUM_EXPRESSION_COEFFS, ext='npz'
        ).to(dtype=DTYPE) # Load with DTYPE
    except Exception as e:
        print(f"Could not load reference SMPL-X model from {MODEL_PATH_HERE}. Error: {e}")
        print("Skipping benchmark.")
        return

    v_template_const = ref_model_params_src.v_template.detach().clone().to(device=device, dtype=DTYPE)
    combined_shapedirs_const = torch.cat(
        [ref_model_params_src.shapedirs, ref_model_params_src.exprdirs], dim=2
    ).detach().clone().to(device=device, dtype=DTYPE)
    
    num_model_vertices = ref_model_params_src.get_num_verts()
    posedirs_ref = ref_model_params_src.posedirs.reshape(num_model_vertices * 3, -1)
    posedirs_const = posedirs_ref.transpose(0,1).contiguous().detach().clone().to(device=device, dtype=DTYPE)
    
    J_regressor_const = ref_model_params_src.J_regressor.detach().clone().to(device=device, dtype=DTYPE)
    parents_const = ref_model_params_src.parents.detach().clone().to(device=device, dtype=torch.int64)
    lbs_weights_const = ref_model_params_src.lbs_weights.detach().clone().to(device=device, dtype=DTYPE)
    
    use_pca_hands_const = False
    lh_components_const = torch.empty(0, device=device, dtype=DTYPE)
    rh_components_const = torch.empty(0, device=device, dtype=DTYPE)

    # Reference model for PyTorch benchmark (move to device)
    ref_model_pytorch = smplx.create(
            MODEL_PATH_HERE, model_type=MODEL_TYPE, gender=GENDER, use_pca=False,
            num_betas=NUM_BETAS, num_expression_coeffs=NUM_EXPRESSION_COEFFS, ext='npz'
        ).to(device=device, dtype=DTYPE)
    ref_model_pytorch.eval()

    results = {"batch_size": [], "pytorch_fwd_time_ms": [], "cuda_fwd_time_ms": [], "speedup_fwd": []}

    for bs in BATCH_SIZES_TO_TEST:
        print(f"--- Benchmarking Batch Size: {bs} ---")
        results["batch_size"].append(bs)
        
        betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose =             get_sample_inputs_bench(bs, device=device)

        # --- PyTorch Reference ---
        pytorch_fwd_times = []
        with torch.no_grad():
            # Warm-up
            for _ in range(NUM_WARMUP_RUNS):
                _ = ref_model_pytorch(betas=betas, expression=expression, global_orient=global_orient,
                                   body_pose=body_pose, jaw_pose=jaw_pose, 
                                   left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
            torch.cuda.synchronize(device=device)
            
            # Timed runs
            for _ in range(NUM_TIMED_RUNS):
                start_time = time.perf_counter()
                _ = ref_model_pytorch(betas=betas, expression=expression, global_orient=global_orient,
                                   body_pose=body_pose, jaw_pose=jaw_pose, 
                                   left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
                torch.cuda.synchronize(device=device)
                end_time = time.perf_counter()
                pytorch_fwd_times.append((end_time - start_time) * 1000) # ms
        
        avg_pytorch_fwd_time = np.mean(pytorch_fwd_times)
        results["pytorch_fwd_time_ms"].append(avg_pytorch_fwd_time)
        print(f"PyTorch Forward: {avg_pytorch_fwd_time:.3f} ms")

        # --- CUDA Implementation ---
        # Using smplx_forward_cuda_no_grad for a fairer forward-only comparison initially
        # Can switch to smplx_cuda_traced if also testing autograd overhead.
        cuda_fwd_times = []
        with torch.no_grad(): # Still use no_grad if calling _no_grad version
             # Warm-up
            for _ in range(NUM_WARMUP_RUNS):
                _ = cuda_smplx_ops.smplx_forward_cuda_no_grad(
                    betas, expression, global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose,
                    v_template_const, combined_shapedirs_const, posedirs_const, J_regressor_const,
                    parents_const, lbs_weights_const, use_pca_hands_const, 
                    lh_components_const, rh_components_const)
            torch.cuda.synchronize(device=device)

            # Timed runs
            for _ in range(NUM_TIMED_RUNS):
                start_time = time.perf_counter()
                _ = cuda_smplx_ops.smplx_forward_cuda_no_grad(
                    betas, expression, global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose,
                    v_template_const, combined_shapedirs_const, posedirs_const, J_regressor_const,
                    parents_const, lbs_weights_const, use_pca_hands_const, 
                    lh_components_const, rh_components_const)
                torch.cuda.synchronize(device=device)
                end_time = time.perf_counter()
                cuda_fwd_times.append((end_time - start_time) * 1000) # ms

        avg_cuda_fwd_time = np.mean(cuda_fwd_times)
        results["cuda_fwd_time_ms"].append(avg_cuda_fwd_time)
        print(f"CUDA Forward:    {avg_cuda_fwd_time:.3f} ms")
        
        speedup = avg_pytorch_fwd_time / avg_cuda_fwd_time if avg_cuda_fwd_time > 0 else float('inf')
        results["speedup_fwd"].append(speedup)
        print(f"Speedup (Forward): {speedup:.2f}x")

    # --- Optional: Plotting ---
    if MATPLOTLIB_AVAILABLE:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Time (ms)', color=color)
        ax1.plot(results["batch_size"], results["pytorch_fwd_time_ms"], color=color, marker='o', linestyle='--', label='PyTorch Fwd')
        ax1.plot(results["batch_size"], results["cuda_fwd_time_ms"], color='tab:blue', marker='o', linestyle='-', label='CUDA Fwd')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Speedup Factor', color=color)
        ax2.plot(results["batch_size"], results["speedup_fwd"], color=color, marker='x', linestyle=':', label='Speedup Fwd')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout() # otherwise the right y-label is slightly clipped
        plt.title('SMPL-X Forward Pass Performance Comparison')
        plt.savefig('smplx_forward_benchmark.png')
        print("Benchmark plot saved to smplx_forward_benchmark.png")
        # plt.show() # Optionally show plot

    return results

if __name__ == "__main__":
    print(f"Looking for SMPL-X models in: {os.path.abspath(MODEL_PATH_HERE)}")
    if not os.path.exists(MODEL_PATH_HERE):
        print(f"Warning: Model path {MODEL_PATH_HERE} does not exist.")
    
    benchmark_results = benchmark_smplx_versions()
    if benchmark_results:
        print("--- Benchmark Summary ---")
        for i in range(len(benchmark_results["batch_size"])):
            bs = benchmark_results["batch_size"][i]
            pt_time = benchmark_results["pytorch_fwd_time_ms"][i]
            cu_time = benchmark_results["cuda_fwd_time_ms"][i]
            sp = benchmark_results["speedup_fwd"][i]
            print(f"Batch: {bs:2d} | PyTorch: {pt_time:7.3f}ms | CUDA: {cu_time:7.3f}ms | Speedup: {sp:5.2f}x")
```
