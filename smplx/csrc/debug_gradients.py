#!/usr/bin/env python3

import torch
import sys
import os

# Add parent directory to path to import the extension and lbs module
csrc_dir = os.path.dirname(os.path.abspath(__file__))
smplx_dir = os.path.dirname(os.path.dirname(csrc_dir))
sys.path.insert(0, smplx_dir)
sys.path.insert(0, os.path.dirname(csrc_dir))

try:
    from smplx.lbs import (
        lbs as lbs_pytorch,
        batch_rigid_transform as batch_rigid_transform_pytorch,
        LBS as LBS_cuda,
        BatchRigidTransform as BatchRigidTransform_cuda,
    )
    from test_lbs_cuda import create_test_data
    print("✓ Successfully imported PyTorch and CUDA LBS functions.")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def compare_gradients(name, torch_grad, cuda_grad):
    """Compare two gradient tensors and print the results."""
    print(f"\n--- Comparing gradients for: {name} ---")
    if torch_grad is None and cuda_grad is None:
        print("✓ Both gradients are None (as expected).")
        return True
    if torch_grad is None or cuda_grad is None:
        print(f"✗ Mismatch: PyTorch grad is {torch_grad is None}, CUDA grad is {cuda_grad is None}")
        return False

    abs_diff = torch.abs(torch_grad - cuda_grad)
    rel_diff = abs_diff / torch.clamp(torch.abs(torch_grad), min=1e-8)

    print(f"  - Shape: {torch_grad.shape}")
    print(f"  - Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"  - Mean absolute difference: {abs_diff.mean().item():.6e}")
    print(f"  - Max relative difference: {rel_diff.max().item():.6e}")
    print(f"  - Mean relative difference: {rel_diff.mean().item():.6e}")

    is_close = torch.allclose(torch_grad, cuda_grad, atol=1e-4, rtol=1e-4)
    if is_close:
        print("✓ Gradients are close.")
    else:
        print("✗ Gradients are NOT close.")
        # print("PyTorch grad:\n", torch_grad)
        # print("CUDA grad:\n", cuda_grad)
    return is_close

def debug_batch_rigid_transform():
    """Debug gradients for batch_rigid_transform."""
    print("\n=============================================")
    print("Debugging batch_rigid_transform gradients")
    print("=============================================")
    
    _, _, rot_mats, joints, parents = create_test_data(device='cuda', dtype=torch.float32)
    
    # --- PyTorch version ---
    rot_mats_torch = rot_mats.clone().detach().requires_grad_(True)
    joints_torch = joints.clone().detach().requires_grad_(True)
    
    posed_joints_torch, rel_transforms_torch = batch_rigid_transform_pytorch(
        rot_mats_torch, joints_torch, parents)
    
    # Create dummy gradients for outputs
    grad_posed_joints = torch.randn_like(posed_joints_torch)
    grad_rel_transforms = torch.randn_like(rel_transforms_torch)
    
    # PyTorch backward pass
    torch.autograd.backward(
        [posed_joints_torch, rel_transforms_torch],
        [grad_posed_joints, grad_rel_transforms]
    )
    
    # --- CUDA version ---
    rot_mats_cuda = rot_mats.clone().detach().requires_grad_(True)
    joints_cuda = joints.clone().detach().requires_grad_(True)
    
    posed_joints_cuda, rel_transforms_cuda = BatchRigidTransform_cuda.apply(
        rot_mats_cuda, joints_cuda, parents)
        
    # CUDA backward pass
    torch.autograd.backward(
        [posed_joints_cuda, rel_transforms_cuda],
        [grad_posed_joints, grad_rel_transforms]
    )
    
    # --- Compare gradients ---
    compare_gradients("rot_mats", rot_mats_torch.grad, rot_mats_cuda.grad)
    compare_gradients("joints", joints_torch.grad, joints_cuda.grad)

def debug_lbs():
    """Debug gradients for LBS."""
    print("\n=============================================")
    print("Debugging LBS gradients")
    print("=============================================")
    
    vertices, weights, _, _, _ = create_test_data(device='cuda', dtype=torch.float32)
    B, V, J = vertices.shape
    transforms = torch.randn(B, J, 4, 4, device='cuda', dtype=torch.float32)

    # --- PyTorch version ---
    vertices_torch = vertices.clone().detach().requires_grad_(True)
    weights_torch = weights.clone().detach().requires_grad_(True)
    transforms_torch = transforms.clone().detach().requires_grad_(True)
    
    B, V, _ = vertices_torch.shape
    J = weights_torch.shape[1]
    
    B, V, _ = vertices_torch.shape
    J = weights_torch.shape[1]
    
    # A simplified PyTorch equivalent for the lbs CUDA kernel
    W = weights_torch.unsqueeze(0).expand(B, -1, -1)
    # Reshape transforms to be (B, J, 16)
    transforms_reshaped = transforms_torch.view(B, J, 16)
    # T = (B, V, J) @ (B, J, 16) -> (B, V, 16)
    T_reshaped = torch.matmul(W, transforms_reshaped)
    # Reshape T to (B, V, 4, 4)
    T = T_reshaped.view(B, V, 4, 4)
    
    homogen_coord = torch.ones(B, V, 1, device=vertices_torch.device, dtype=vertices_torch.dtype)
    v_homo = torch.cat([vertices_torch, homogen_coord], dim=2).unsqueeze(-1)
    
    posed_v_homo = torch.matmul(T, v_homo)
    posed_vertices_torch = posed_v_homo[:, :, :3, 0]

    grad_posed_vertices = torch.randn_like(posed_vertices_torch)
    posed_vertices_torch.backward(grad_posed_vertices)

    # --- CUDA version ---
    vertices_cuda = vertices.clone().detach().requires_grad_(True)
    weights_cuda = weights.clone().detach().requires_grad_(True)
    transforms_cuda = transforms.clone().detach().requires_grad_(True)
    
    posed_vertices_cuda = LBS_cuda.apply(vertices_cuda, weights_cuda, transforms_cuda)
    posed_vertices_cuda.backward(grad_posed_vertices)
    
    # --- Compare gradients ---
    compare_gradients("vertices", vertices_torch.grad, vertices_cuda.grad)
    compare_gradients("weights", weights_torch.grad, weights_cuda.grad)
    compare_gradients("transforms", transforms_torch.grad, transforms_cuda.grad)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        sys.exit(1)
    
    debug_batch_rigid_transform()
    debug_lbs()
