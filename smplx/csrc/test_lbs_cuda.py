#!/usr/bin/env python3

import torch
from torch.autograd import gradcheck
import numpy as np
import time
import sys
import os

# Add parent directory to path to import the extension and lbs module
csrc_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(csrc_dir))
smplx_dir = os.path.dirname(os.path.dirname(csrc_dir))
sys.path.insert(0, smplx_dir)

try:
    import lbs_cuda_ext
    # The custom autograd Functions are defined in the lbs python module
    from smplx.lbs import LBS, BatchRigidTransform
    print("✓ Successfully imported lbs_cuda_ext and smplx.lbs")
except ImportError as e:
    print(f"✗ Failed to import extension or smplx.lbs: {e}")
    print("Please run: ./build_lbs.sh from the csrc directory")
    sys.exit(1)

def create_test_data(B=2, V=10, J=4, device='cuda', dtype=torch.float32):
    """Create test data for LBS. Using smaller V and J for faster gradcheck."""
    
    vertices = torch.randn(B, V, 3, device=device, dtype=dtype)
    weights = torch.rand(V, J, device=device, dtype=dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    rot_mats = torch.randn(B, J, 3, 3, device=device, dtype=dtype)
    U, _, Vt = torch.linalg.svd(rot_mats)
    rot_mats = torch.matmul(U, Vt)
    
    joints = torch.randn(B, J, 3, device=device, dtype=dtype)
    
    parents = torch.arange(J, device='cpu', dtype=torch.int32) - 1
    parents[0] = -1
    
    return vertices, weights, rot_mats, joints, parents.to(device)

def test_batch_rigid_transform():
    """Test batch rigid transform kernel forward pass"""
    print("\nTesting batch_rigid_transform...")
    
    _, _, rot_mats, joints, parents = create_test_data(B=2, J=24)
    
    try:
        posed_joints, rel_transforms, _ = lbs_cuda_ext.batch_rigid_transform(
            rot_mats, joints, parents.float())
        print(f"✓ batch_rigid_transform output shapes: {posed_joints.shape}, {rel_transforms.shape}")
        
        expected_shape_posed = (rot_mats.shape[0], rot_mats.shape[1], 3)
        expected_shape_rel = (rot_mats.shape[0], rot_mats.shape[1], 4, 4)
        
        if posed_joints.shape == expected_shape_posed and rel_transforms.shape == expected_shape_rel:
            print("✓ Output shapes correct")
        else:
            print(f"✗ Wrong output shapes.")
            return False
            
        bottom_row = rel_transforms[:, :, 3, :]
        expected_bottom = torch.tensor([0, 0, 0, 1], device=rot_mats.device, dtype=torch.float32)
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
    """Test LBS kernel forward pass"""
    print("\nTesting lbs...")
    
    vertices, weights, _, _, _ = create_test_data(V=1000, J=24)
    B, V, J = vertices.shape[0], vertices.shape[1], weights.shape[1]
    
    transforms = torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, J, 1, 1)
    
    try:
        posed_vertices = lbs_cuda_ext.lbs(vertices, weights, transforms)
        print(f"✓ lbs output shape: {posed_vertices.shape}")
        
        if posed_vertices.shape == vertices.shape:
            print("✓ Output shape correct")
        else:
            print(f"✗ Wrong output shape. Expected {vertices.shape}, got {posed_vertices.shape}")
            return False
            
        if torch.allclose(posed_vertices, vertices, atol=1e-4):
            print("✓ Identity transform test passed")
        else:
            print("✗ Identity transform test failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ lbs failed: {e}")
        return False

def test_lbs_backward():
    """Test LBS backward pass using gradcheck"""
    print("\nTesting lbs_backward (gradient check)...")
    
    # gradcheck needs double precision and CPU tensors
    vertices, weights, _, _, _ = create_test_data(device='cpu', dtype=torch.double)
    transforms = torch.randn(vertices.shape[0], weights.shape[1], 4, 4, device='cpu', dtype=torch.double)
    
    vertices.requires_grad = True
    weights.requires_grad = True
    transforms.requires_grad = True
    
    def lbs_wrapper(v, w, t):
        # Cast to float for CUDA extension, then back to double for gradcheck
        output = LBS.apply(v.float().cuda(), w.float().cuda(), t.float().cuda())
        return output.cpu().double()

    try:
        is_correct = gradcheck(lbs_wrapper, (vertices, weights, transforms), eps=1e-6, atol=1e-4)
        if is_correct:
            print("✓ LBS backward pass is correct")
            return True
        else:
            print("✗ LBS backward pass is incorrect")
            return False
    except Exception as e:
        print(f"✗ LBS backward gradcheck failed: {e}")
        return False

def test_batch_rigid_transform_backward():
    """Test batch_rigid_transform backward pass using gradcheck"""
    print("\nTesting batch_rigid_transform_backward (gradient check)...")
    
    # gradcheck needs double precision and CPU tensors
    _, _, rot_mats, joints, parents = create_test_data(device='cpu', dtype=torch.double)
    
    rot_mats.requires_grad = True
    joints.requires_grad = True
    
    def brt_wrapper_posed_joints(r, j):
        r_f = r.float().cuda()
        j_f = j.float().cuda()
        p_cuda = parents.cuda()
        posed_joints_f, _ = BatchRigidTransform.apply(r_f, j_f, p_cuda)
        return posed_joints_f.cpu().double()
        
    def brt_wrapper_rel_transforms(r, j):
        r_f = r.float().cuda()
        j_f = j.float().cuda()
        p_cuda = parents.cuda()
        _, rel_transforms_f = BatchRigidTransform.apply(r_f, j_f, p_cuda)
        return rel_transforms_f.cpu().double()

    try:
        print("  - Checking gradients for posed_joints...")
        is_correct_posed = gradcheck(brt_wrapper_posed_joints, (rot_mats, joints), eps=1e-6, atol=1e-4)
        if not is_correct_posed:
            print("✗ Gradients for posed_joints are incorrect")
            return False
        print("✓ Gradients for posed_joints are correct")

        print("  - Checking gradients for rel_transforms...")
        is_correct_rel = gradcheck(brt_wrapper_rel_transforms, (rot_mats, joints), eps=1e-6, atol=1e-4)
        if not is_correct_rel:
            print("✗ Gradients for rel_transforms are incorrect")
            return False
        print("✓ Gradients for rel_transforms are correct")
            
        return True
    except Exception as e:
        print(f"✗ batch_rigid_transform backward gradcheck failed: {e}")
        return False

def benchmark_lbs():
    """Benchmark LBS performance"""
    print("\nBenchmarking LBS performance...")
    
    vertices, weights, rot_mats, joints, parents = create_test_data(B=16, V=6890, J=24)
    
    # Warmup
    for _ in range(5):
        lbs_cuda_ext.lbs_forward(vertices, weights, rot_mats, joints, parents.float())
    
    torch.cuda.synchronize()
    
    num_runs = 100
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        lbs_cuda_ext.lbs_forward(vertices, weights, rot_mats, joints, parents.float())
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs * 1000
    print(f"✓ Average LBS time: {avg_time:.2f}ms for {vertices.shape[0]} batches, {vertices.shape[1]} vertices, {weights.shape[1]} joints")

def main():
    print("LBS CUDA Extension Test")
    print("======================")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False
    
    print(f"✓ CUDA available, using device: {torch.cuda.get_device_name()}")
    
    tests = [
        test_batch_rigid_transform,
        test_lbs,
        test_lbs_backward,
        test_batch_rigid_transform_backward,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
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