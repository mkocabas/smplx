#pragma once

#include <torch/extension.h>
#include <vector> // Required for parents

// Outputs two tensors: posed_joints and A_global
std::vector<torch::Tensor> batch_rigid_transform_cuda(
    torch::Tensor rot_mats,    // (batch_size, num_joints, 3, 3)
    torch::Tensor joints,      // (batch_size, num_joints, 3)
    torch::Tensor parents      // (num_joints) --> int64 type
);

// Backward function declaration
std::vector<torch::Tensor> batch_rigid_transform_backward_cuda(
    torch::Tensor grad_posed_joints,    // (B, J, 3)
    torch::Tensor grad_A_global,        // (B, J, 4, 4)
    torch::Tensor rot_mats_fwd,         // (B, J, 3, 3)
    torch::Tensor joints_fwd,           // (B, J, 3) (J_shaped)
    torch::Tensor parents_fwd,          // (J) int64
    // Forward outputs that might be useful:
    torch::Tensor posed_joints_fwd,     // (B, J, 3)
    torch::Tensor A_global_fwd          // (B, J, 4, 4)
    // We also need G_fwd (transform_chain_b in fwd kernel). Recompute or pass if possible.
    // For now, let's try to recompute G_fwd from A_global_fwd and joints_fwd.
);
