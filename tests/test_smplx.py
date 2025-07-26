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

def generate_test_params(batch_size: int, device: str):
    """Generate consistent test parameters"""
    torch.manual_seed(42)  # For reproducible results
    return {
        'body_pose': torch.rand(batch_size, 63).to(device) * 0.5,
        'global_orient': torch.rand(batch_size, 3).to(device),
        'betas': torch.rand(batch_size, 10).to(device),
        'left_hand_pose': torch.rand(batch_size, 45).to(device),
        'right_hand_pose': torch.rand(batch_size, 45).to(device)
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

def benchmark_model(model, params: Dict, model_name: str, num_warmup: int = 5, num_runs: int = 20):
    """Benchmark model runtime"""
    device = next(model.parameters()).device
    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**params)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
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

def test_smplx_models(smplx_path: str, device: str = 'cuda', batch_sizes: List[int] = [2, 16, 32, 64, 128]):
    """Test different SMPLX model variants across various batch sizes"""
    results = {
        'batch_sizes': batch_sizes,
        'models': {},
        'correctness': {}
    }
    
    print(f"Testing SMPLX models on {device}")
    print("=" * 50)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 30)
        
        # Create models for this batch size
        models = create_smplx_models(smplx_path, device, batch_size)
        params = generate_test_params(batch_size, device)
        
        # Run baseline first
        baseline_output = models['baseline'](**params)
        
        batch_results = {}
        for model_name, model in models.items():
            if model_name == 'baseline':
                # Benchmark baseline
                benchmark_result = benchmark_model(model, params, model_name)
                batch_results[model_name] = benchmark_result
                print(f"{model_name:12}: {benchmark_result['avg_time_ms']:.2f}±{benchmark_result['std_time_ms']:.2f}ms")
            else:
                # Test output correctness
                test_output = model(**params)
                is_correct = verify_outputs(baseline_output, test_output, model_name)
                
                if model_name not in results['correctness']:
                    results['correctness'][model_name] = []
                results['correctness'][model_name].append(is_correct)
                
                # Benchmark
                benchmark_result = benchmark_model(model, params, model_name)
                batch_results[model_name] = benchmark_result
                print(f"{model_name:12}: {benchmark_result['avg_time_ms']:.2f}±{benchmark_result['std_time_ms']:.2f}ms")
        
        # Store results for this batch size
        for model_name, result in batch_results.items():
            if model_name not in results['models']:
                results['models'][model_name] = {'times': []}
            results['models'][model_name]['times'].append(result['avg_time_ms'])
        
        # Clean up GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    return results

def plot_benchmark_results(results: Dict, save_path: str = 'smplx_benchmark.png'):
    """Plot benchmark results and save as PNG"""
    plt.figure(figsize=(10, 6))
    
    batch_sizes = results['batch_sizes']
    
    # Plot runtime
    for model_name, data in results['models'].items():
        plt.plot(batch_sizes, data['times'], marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Runtime (ms)')
    plt.title('SMPLX Runtime Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nBenchmark plot saved to: {save_path}")

# Run the comprehensive test
if __name__ == "__main__":
    smplx_path = '/home/muhammed/projects/camera_motion/data/models/SMPLX/neutral/SMPLX_neutral.npz'
    device = 'cuda'
    batch_sizes = [2, 16, 32, 64, 128, 1024]  # Reduced for faster testing
    
    results = test_smplx_models(smplx_path, device, batch_sizes)
    plot_benchmark_results(results)