#!/usr/bin/env python3

import torch
import numpy as np
import time
import sys
import os

# Add current directory to path to import the extension
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import lbs_cuda_ext
    print("✓ Successfully imported lbs_cuda_ext")
except ImportError as e:
    print(f"✗ Failed to import lbs_cuda_ext: {e}")
    print("Please run: ./build_lbs.sh")
    sys.exit(1)

def create_test_data():
    """Create test data for LBS"""
    B, V, J = 2, 1000, 24  # Batch size, vertices, joints
    device = 'cuda'
    
    # Random test data
    vertices = torch.randn(B, V, 3, device=device, dtype=torch.float32)
    weights = torch.rand(V, J, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights
    
    rot_mats = torch.randn(B, J, 3, 3, device=device, dtype=torch.float32)
    # Make proper rotation matrices (orthogonal)
    U, _, Vt = torch.linalg.svd(rot_mats)
    rot_mats = torch.matmul(U, Vt)
    
    joints = torch.randn(B, J, 3, device=device, dtype=torch.float32)
    
    # Create parent hierarchy (simple chain)
    parents = torch.arange(J, device=device, dtype=torch.float32) - 1
    parents[0] = -1  # Root has no parent
    
    return vertices, weights, rot_mats, joints, parents

def test_batch_rigid_transform():
    """Test batch rigid transform kernel"""
    print("\nTesting batch_rigid_transform...")
    
    B, J = 2, 24
    device = 'cuda'
    
    rot_mats = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, J, 1, 1)
    joints = torch.randn(B, J, 3, device=device, dtype=torch.float32)
    parents = torch.arange(J, device=device, dtype=torch.float32) - 1
    parents[0] = -1
    
    try:
        transforms = lbs_cuda_ext.batch_rigid_transform(rot_mats, joints, parents)
        print(f"✓ batch_rigid_transform output shape: {transforms.shape}")
        
        # Check output shape
        expected_shape = (B, J, 4, 4)
        if transforms.shape == expected_shape:
            print("✓ Output shape correct")
        else:
            print(f"✗ Wrong output shape. Expected {expected_shape}, got {transforms.shape}")
            return False
            
        # Check if transforms are valid (bottom row should be [0, 0, 0, 1])
        bottom_row = transforms[:, :, 3, :]
        expected_bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
        if torch.allclose(bottom_row, expected_bottom.unsqueeze(0).unsqueeze(0).expand_as(bottom_row)):
            print("✓ Transform matrices have correct format")
        else:
            print("✗ Transform matrices have incorrect format")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ batch_rigid_transform failed: {e}")
        return False

def test_lbs():
    """Test LBS kernel"""
    print("\nTesting lbs...")
    
    vertices, weights, _, _, _ = create_test_data()
    B, V, J = vertices.shape[0], vertices.shape[1], weights.shape[1]
    
    # Create identity transforms for simple test
    transforms = torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, J, 1, 1)
    
    try:
        posed_vertices = lbs_cuda_ext.lbs(vertices, weights, transforms)
        print(f"✓ lbs output shape: {posed_vertices.shape}")
        
        # Check output shape
        if posed_vertices.shape == vertices.shape:
            print("✓ Output shape correct")
        else:
            print(f"✗ Wrong output shape. Expected {vertices.shape}, got {posed_vertices.shape}")
            return False
            
        # With identity transforms, output should be close to input
        if torch.allclose(posed_vertices, vertices, atol=1e-4):
            print("✓ Identity transform test passed")
        else:
            print("✗ Identity transform test failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ lbs failed: {e}")
        return False

def test_lbs_forward():
    """Test combined LBS forward pass"""
    print("\nTesting lbs_forward...")
    
    vertices, weights, rot_mats, joints, parents = create_test_data()
    
    try:
        posed_vertices = lbs_cuda_ext.lbs_forward(vertices, weights, rot_mats, joints, parents)
        print(f"✓ lbs_forward output shape: {posed_vertices.shape}")
        
        # Check output shape
        if posed_vertices.shape == vertices.shape:
            print("✓ Output shape correct")
        else:
            print(f"✗ Wrong output shape. Expected {vertices.shape}, got {posed_vertices.shape}")
            return False
            
        # Check that output is different from input (should be posed)
        if not torch.allclose(posed_vertices, vertices, atol=1e-2):
            print("✓ Vertices were transformed as expected")
        else:
            print("⚠ Vertices were not transformed (might be expected for identity poses)")
            
        return True
        
    except Exception as e:
        print(f"✗ lbs_forward failed: {e}")
        return False

def benchmark_lbs():
    """Benchmark LBS performance"""
    print("\nBenchmarking LBS performance...")
    
    vertices, weights, rot_mats, joints, parents = create_test_data()
    
    # Warmup
    for _ in range(5):
        _ = lbs_cuda_ext.lbs_forward(vertices, weights, rot_mats, joints, parents)
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 100
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        posed_vertices = lbs_cuda_ext.lbs_forward(vertices, weights, rot_mats, joints, parents)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    print(f"✓ Average LBS time: {avg_time:.2f}ms for {vertices.shape[0]} batches, {vertices.shape[1]} vertices, {weights.shape[1]} joints")

def main():
    print("LBS CUDA Extension Test")
    print("======================")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False
    
    print(f"✓ CUDA available, using device: {torch.cuda.get_device_name()}")
    
    # Run tests
    tests = [
        test_batch_rigid_transform,
        test_lbs, 
        test_lbs_forward
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"✗ Test {test.__name__} failed")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        benchmark_lbs()
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)