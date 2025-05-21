#pragma once

#include <torch/extension.h>
#include <vector>

// Returns a vector of tensors representing gradients of inputs to smplx_forward_cuda
// Order must match the inputs of smplx_forward_cuda for which gradients are required.
// E.g., {grad_betas, grad_expression, grad_global_orient, grad_body_pose, ... , grad_lbs_weights}
// Gradients for fixed model params like v_template, shapedirs, J_regressor, parents, lbs_weights
// can also be returned if they are learnable (often they are fixed).
std::vector<torch::Tensor> smplx_backward_cuda(
    // Gradients from Python autograd context
    torch::Tensor grad_final_vertices, // dL/dfinal_vertices
    torch::Tensor grad_posed_joints,   // dL/dposed_joints

    // Saved tensors from forward pass (passed from Python context)
    // Inputs to smplx_forward_cuda
    torch::Tensor betas_fwd,
    torch::Tensor expression_fwd,
    torch::Tensor global_orient_fwd,
    torch::Tensor body_pose_fwd,
    torch::Tensor left_hand_pose_fwd,
    torch::Tensor right_hand_pose_fwd,
    torch::Tensor jaw_pose_fwd,
    
    torch::Tensor v_template_fwd,
    torch::Tensor shapedirs_fwd, // Combined shape + expression
    torch::Tensor posedirs_fwd,
    torch::Tensor J_regressor_fwd,
    torch::Tensor parents_fwd,
    torch::Tensor lbs_weights_fwd,
    
    bool use_pca_hands_fwd,
    torch::Tensor left_hand_components_fwd, // If used
    torch::Tensor right_hand_components_fwd, // If used

    // Intermediates from forward pass (if saved and needed, otherwise recompute)
    // These are outputs of the individual CUDA functions from smplx_forward_cuda
    torch::Tensor v_shaped_fwd,
    torch::Tensor J_shaped_fwd,
    torch::Tensor full_pose_axis_angle_fwd, // Concatenated pose before rodrigues
    torch::Tensor rot_mats_fwd,
    torch::Tensor posed_joints_fwd, // This is one of the outputs, its grad is grad_posed_joints
    torch::Tensor A_global_fwd,
    torch::Tensor v_posed_fwd // v_shaped + pose_offsets
);
