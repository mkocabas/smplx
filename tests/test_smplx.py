import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from smplx import SMPLX


# INFO: rest of the code is a just a sample sample smplx forward call, use this as a reference
smplx_path = '/home/muhammed/projects/camera_motion/data/models/SMPLX/neutral/SMPLX_neutral.npz'

def sanity_check():
    device = 'cuda'

    B = 10

    smplx = SMPLX(
        smplx_path,
        use_pca=False, 
        flat_hand_mean=True, 
        num_betas=10,
        batch_size=B,
    ).to(device)

    params = {
        'body_pose': torch.rand(B, 63).to(device) * 0.5,
        'global_orient': torch.rand(B, 3).to(device),
        'betas': torch.rand(B, 10).to(device),
        'left_hand_pose': torch.rand(B, 45).to(device),
        'right_hand_pose': torch.rand(B, 45).to(device)
    }

    smplx_output = smplx(**params)

    print(f"Vertices shape: {smplx_output.vertices.shape}")
    print(f"Joints shape: {smplx_output.joints.shape}")

    def save_obj(vertices, faces, filename):
        """Save mesh as OBJ file"""
        with open(filename, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Get vertices and faces for the first mesh in the batch
    vertices = smplx_output.vertices[0].detach().cpu().numpy()
    faces = smplx.faces

    # Save the mesh as OBJ file
    save_obj(vertices, faces, 'output_mesh.obj')
    print(f"Saved mesh to output_mesh.obj")

import time
import gc
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Callable

def create_smplx_models(smplx_path: str, device: str, batch_size: int):
    """Create different SMPLX model variants for testing"""
    models = {}
    
    # Baseline model
    baseline = SMPLX(
        smplx_path,
        use_pca=False, 
        flat_hand_mean=True, 
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    models['baseline'] = baseline

    
    # Torch compiled model
    compiled_model = SMPLX(
        smplx_path,
        use_pca=False, 
        flat_hand_mean=True, 
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    compiled = torch.compile(compiled_model, mode='default')
    models['compiled'] = compiled
    
    # CUDA LBS model
    cuda_model = SMPLX(
        smplx_path,
        use_pca=False, 
        flat_hand_mean=True, 
        num_betas=10,
        batch_size=batch_size,
        use_cuda_lbs=True,
    ).to(device)
    models['cuda_lbs'] = cuda_model
    
    # Sanity check model (same as baseline)
    sanity = SMPLX(
        smplx_path,
        use_pca=False, 
        flat_hand_mean=True, 
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    models['sanity'] = sanity
    
    return models

def generate_test_params(batch_size: int, device: str, requires_grad: bool = False):
    """Generate consistent test parameters"""
    torch.manual_seed(42)  # For reproducible results
    return {
        'body_pose': torch.rand(batch_size, 63, device=device, requires_grad=requires_grad) * 0.5,
        'global_orient': torch.rand(batch_size, 3, device=device, requires_grad=requires_grad),
        'betas': torch.rand(batch_size, 10, device=device, requires_grad=requires_grad),
        'left_hand_pose': torch.rand(batch_size, 45, device=device, requires_grad=requires_grad),
        'right_hand_pose': torch.rand(batch_size, 45, device=device, requires_grad=requires_grad)
    }

def verify_outputs(baseline_output, test_output, model_name: str, tolerance: float = 1e-4):
    """Verify that model outputs match baseline within tolerance"""
    vertices_match = torch.allclose(baseline_output.vertices, test_output.vertices, atol=tolerance)
    joints_match = torch.allclose(baseline_output.joints, test_output.joints, atol=tolerance)
    
    if vertices_match and joints_match:
        print(f"✓ {model_name}: Output matches baseline (tolerance: {tolerance})")
        return True
    else:
        max_v_diff = torch.max(torch.abs(baseline_output.vertices - test_output.vertices)).item()
        max_j_diff = torch.max(torch.abs(baseline_output.joints - test_output.joints)).item()
        print(f"✗ {model_name}: Output differs from baseline")
        print(f"  Max vertex diff: {max_v_diff:.6f}")
        print(f"  Max joint diff: {max_j_diff:.6f}")
        return False

def verify_gradients(baseline_grads, test_grads, model_name: str, tolerance: float = 1e-4):
    """Verify that gradients match baseline within tolerance"""
    all_match = True
    
    for param_name in baseline_grads.keys():
        if param_name in test_grads:
            if baseline_grads[param_name] is None or test_grads[param_name] is None:
                match = baseline_grads[param_name] is None and test_grads[param_name] is None
            else:
                match = torch.allclose(baseline_grads[param_name], test_grads[param_name], atol=tolerance)
            
            if not match:
                all_match = False
                if baseline_grads[param_name] is not None and test_grads[param_name] is not None:
                    max_diff = torch.max(torch.abs(baseline_grads[param_name] - test_grads[param_name])).item()
                    print(f"✗ {model_name}: {param_name} gradient differs (max diff: {max_diff:.6f})")
                else:
                    print(f"✗ {model_name}: {param_name} gradient mismatch (None vs tensor)")
    
    if all_match:
        print(f"✓ {model_name}: Gradients match baseline (tolerance: {tolerance})")
    
    return all_match

def benchmark_model(model, params: Dict, model_name: str, num_warmup: int = 5, num_runs: int = 20, test_backward: bool = False):
    """Benchmark model runtime for forward and optionally backward pass"""
    device = next(model.parameters()).device
    
    # Warmup runs
    for _ in range(num_warmup):
        if test_backward:
            # Create fresh parameters for each run
            fresh_params = {}
            for name, param in params.items():
                fresh_params[name] = param.detach().clone().requires_grad_(True)
            
            output = model(**fresh_params)
            loss = output.vertices.sum() + output.joints.sum()
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(**params)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        if test_backward:
            # Create fresh parameters for each run
            fresh_params = {}
            for name, param in params.items():
                fresh_params[name] = param.detach().clone().requires_grad_(True)
            
            output = model(**fresh_params)
            loss = output.vertices.sum() + output.joints.sum()
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(**params)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time
    }

def compute_gradients(model, params: Dict):
    """Compute gradients for all parameters"""
    # Create fresh parameters with requires_grad=True
    fresh_params = {}
    for name, param in params.items():
        fresh_params[name] = param.detach().clone().requires_grad_(True)
    
    model.zero_grad()
    output = model(**fresh_params)
    loss = output.vertices.sum() + output.joints.sum()
    loss.backward()
    
    gradients = {}
    for name, param in fresh_params.items():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
        else:
            gradients[name] = None
    
    return gradients

def test_smplx_models(smplx_path: str, device: str = 'cuda', batch_sizes: List[int] = [2, 16, 32, 64, 128], test_backward: bool = True):
    """Test different SMPLX model variants across various batch sizes"""
    results = {
        'batch_sizes': batch_sizes,
        'models': {},
        'backward_models': {},
        'correctness': {},
        'backward_correctness': {}
    }
    
    print(f"Testing SMPLX models on {device}")
    print("=" * 50)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 30)
        
        # Create models for this batch size
        models = create_smplx_models(smplx_path, device, batch_size)
        params = generate_test_params(batch_size, device)
        params_grad = generate_test_params(batch_size, device, requires_grad=False) if test_backward else None
        
        # Run baseline first
        baseline_output = models['baseline'](**params)
        baseline_grads = None
        if test_backward:
            baseline_grads = compute_gradients(models['baseline'], params_grad)
        
        batch_results = {}
        batch_backward_results = {}
        
        for model_name, model in models.items():
            if model_name == 'baseline':
                # Benchmark baseline forward
                benchmark_result = benchmark_model(model, params, model_name)
                batch_results[model_name] = benchmark_result
                print(f"{model_name:12} (fwd): {benchmark_result['avg_time_ms']:.2f}±{benchmark_result['std_time_ms']:.2f}ms")
                
                # Benchmark baseline backward
                if test_backward:
                    backward_result = benchmark_model(model, params_grad, model_name, test_backward=True)
                    batch_backward_results[model_name] = backward_result
                    print(f"{model_name:12} (bwd): {backward_result['avg_time_ms']:.2f}±{backward_result['std_time_ms']:.2f}ms")
            else:
                # Test forward output correctness
                test_output = model(**params)
                is_correct = verify_outputs(baseline_output, test_output, model_name)
                
                if model_name not in results['correctness']:
                    results['correctness'][model_name] = []
                results['correctness'][model_name].append(is_correct)
                
                # Test backward correctness
                if test_backward:
                    test_grads = compute_gradients(model, params_grad)
                    is_backward_correct = verify_gradients(baseline_grads, test_grads, model_name)
                    
                    if model_name not in results['backward_correctness']:
                        results['backward_correctness'][model_name] = []
                    results['backward_correctness'][model_name].append(is_backward_correct)
                
                # Benchmark forward
                benchmark_result = benchmark_model(model, params, model_name)
                batch_results[model_name] = benchmark_result
                print(f"{model_name:12} (fwd): {benchmark_result['avg_time_ms']:.2f}±{benchmark_result['std_time_ms']:.2f}ms")
                
                # Benchmark backward
                if test_backward:
                    backward_result = benchmark_model(model, params_grad, model_name, test_backward=True)
                    batch_backward_results[model_name] = backward_result
                    print(f"{model_name:12} (bwd): {backward_result['avg_time_ms']:.2f}±{backward_result['std_time_ms']:.2f}ms")
        
        # Store results for this batch size
        for model_name, result in batch_results.items():
            if model_name not in results['models']:
                results['models'][model_name] = {'times': []}
            results['models'][model_name]['times'].append(result['avg_time_ms'])
        
        # Store backward results
        if test_backward:
            for model_name, result in batch_backward_results.items():
                if model_name not in results['backward_models']:
                    results['backward_models'][model_name] = {'times': []}
                results['backward_models'][model_name]['times'].append(result['avg_time_ms'])
        
        # Clean up GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print("\nForward Pass Correctness:")
    for model_name, correctness_list in results['correctness'].items():
        all_correct = all(correctness_list)
        print(f"  {model_name}: {'✓ ALL BATCHES PASS' if all_correct else '✗ SOME BATCHES FAIL'}")
    
    if test_backward and 'backward_correctness' in results:
        print("\nBackward Pass Correctness:")
        for model_name, correctness_list in results['backward_correctness'].items():
            all_correct = all(correctness_list)
            print(f"  {model_name}: {'✓ ALL BATCHES PASS' if all_correct else '✗ SOME BATCHES FAIL'}")
    
    return results

def plot_benchmark_results(results: Dict, save_path: str = 'smplx_benchmark.png'):
    """Plot benchmark results and save as PNG"""
    has_backward = 'backward_models' in results and len(results['backward_models']) > 0
    
    if has_backward:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        _, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    batch_sizes = results['batch_sizes']
    
    # Plot forward pass runtime
    for model_name, data in results['models'].items():
        ax1.plot(batch_sizes, data['times'], marker='o', label=f'{model_name}', linewidth=2)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Runtime (ms)')
    ax1.set_title('SMPLX Forward Pass Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot backward pass runtime if available
    if has_backward:
        for model_name, data in results['backward_models'].items():
            ax2.plot(batch_sizes, data['times'], marker='s', label=f'{model_name}', linewidth=2)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Runtime (ms)')
        ax2.set_title('SMPLX Backward Pass Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nBenchmark plot saved to: {save_path}")

# Run the comprehensive test
if __name__ == "__main__":
    smplx_path = '/home/muhammed/projects/camera_motion/data/models/SMPLX/neutral/SMPLX_neutral.npz'
    device = 'cuda'
    batch_sizes = [2, 16, 32, 64, 128, 1024]  # Reduced for faster testing
    
    print("Testing forward and backward pass correctness and performance...")
    results = test_smplx_models(smplx_path, device, batch_sizes, test_backward=True)
    plot_benchmark_results(results)
    
    # Print correctness summary
    print("\n" + "=" * 50)
    print("CORRECTNESS SUMMARY")
    print("=" * 50)
    
    print("\nForward Pass Correctness:")
    for model_name, correctness_list in results['correctness'].items():
        all_correct = all(correctness_list)
        print(f"  {model_name}: {'✓ PASS' if all_correct else '✗ FAIL'}")
    
    if 'backward_correctness' in results:
        print("\nBackward Pass Correctness:")
        for model_name, correctness_list in results['backward_correctness'].items():
            all_correct = all(correctness_list)
            print(f"  {model_name}: {'✓ PASS' if all_correct else '✗ FAIL'}")