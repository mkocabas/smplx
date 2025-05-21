#pragma once

#include <torch/extension.h>
#include <vector> // Added for std::vector return types

// Combined function that will call the necessary kernels
torch::Tensor skinning_cuda(
    torch::Tensor v_shaped,          // (batch_size, num_vertices, 3) - from blend_shapes
    torch::Tensor rot_mats,          // (batch_size, num_joints, 3, 3) - full rotation matrices
    torch::Tensor posedirs,          // ( (num_joints-1)*9, num_vertices*3 )
    torch::Tensor lbs_weights,       // (num_vertices, num_joints)
    torch::Tensor A_global           // (batch_size, num_joints, 4, 4) - from batch_rigid_transform
);

// Individual helper for pose_offsets (can be static or inlined in .cu if not exposed)
torch::Tensor calculate_pose_offsets_cuda(
    torch::Tensor rot_mats,  // (batch_size, num_joints, 3, 3)
    torch::Tensor posedirs   // ( (num_joints-1)*9, num_vertices*3 )
);

// Backward function declaration for the main skinning_cuda orchestrator
std::vector<torch::Tensor> skinning_backward_cuda(
    torch::Tensor grad_verts,         // (B, V, 3)
    // Forward inputs needed for various gradient calculations:
    torch::Tensor v_shaped_fwd,       // (B, V, 3)
    torch::Tensor rot_mats_fwd,       // (B, J, 3, 3) - for pose_offsets_bwd
    torch::Tensor posedirs_fwd,       // (P, V*3) - for pose_offsets_bwd
    torch::Tensor lbs_weights_fwd,    // (V, J) - for skinning_transform_bwd
    torch::Tensor A_global_fwd,       // (B, J, 4, 4) - for skinning_transform_bwd
    // Forward intermediate v_posed can be recomputed or taken as input if available
    torch::Tensor v_posed_fwd         // (B, V, 3) = v_shaped_fwd + pose_offsets_fwd
);

// Backward for calculate_pose_offsets (if called separately or by skinning_backward_cuda)
std::vector<torch::Tensor> calculate_pose_offsets_backward_cuda(
    torch::Tensor grad_pose_offsets, // (B, V, 3)
    torch::Tensor rot_mats_fwd,      // (B, J, 3, 3)
    torch::Tensor posedirs_fwd       // (P, V*3)
);

// Backward for the core skinning transformation (v_posed -> verts)
std::vector<torch::Tensor> skinning_transform_backward_cuda(
    torch::Tensor grad_verts,        // (B, V, 3)
    torch::Tensor v_posed_fwd,       // (B, V, 3)
    torch::Tensor lbs_weights_fwd,   // (V, J)
    torch::Tensor A_global_fwd       // (B, J, 4, 4)
);
