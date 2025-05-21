#pragma once

#include <torch/extension.h>
#include <vector>

// Returns a vector of tensors: [final_vertices, posed_joints]
std::vector<torch::Tensor> smplx_forward_cuda(
    // Input parameters
    torch::Tensor betas,              // (batch_size, num_betas)
    torch::Tensor expression,         // (batch_size, num_expression_coeffs) - specific to SMPLX
    torch::Tensor global_orient,      // (batch_size, 3) - axis-angle
    torch::Tensor body_pose,          // (batch_size, num_body_joints * 3) - axis-angle
    torch::Tensor left_hand_pose,     // (batch_size, num_hand_joints * 3 or pca_comps)
    torch::Tensor right_hand_pose,    // (batch_size, num_hand_joints * 3 or pca_comps)
    torch::Tensor jaw_pose,           // (batch_size, 3) - axis-angle (SMPLX)
    // TODO: Add eye poses if they are part of the kinematic chain affecting vertices

    // Model constants (pre-loaded on GPU)
    torch::Tensor v_template,         // (num_vertices, 3)
    torch::Tensor shapedirs,          // (num_vertices, 3, num_shape_coeffs + num_expr_coeffs)
                                      // Note: SMPLX uses betas for shape and expression for expression_coeffs.
                                      // shapedirs here should contain both shape and expression blendshapes.
                                      // The calling Python code will need to concatenate these.
    torch::Tensor posedirs,           // ( (num_joints-1)*9, num_vertices*3 )
    torch::Tensor J_regressor,        // (num_joints, num_vertices)
    torch::Tensor parents,            // (num_joints) - int64, CPU or GPU (kernel handles copy if needed)
    torch::Tensor lbs_weights,        // (num_vertices, num_joints)
    
    // Configuration (can be derived or passed)
    bool use_pca_hands = false,       // If true, hand poses are PCA and need conversion
    torch::Tensor left_hand_components = torch::empty({0}), // PCA components if use_pca_hands
    torch::Tensor right_hand_components = torch::empty({0}) // PCA components if use_pca_hands
    // Note: For a generic SMPLX, hand/jaw/eye poses are part of a single 'full_pose' tensor.
    // The input pose parameters here are split for clarity but will be concatenated.
    // The number of joints for posedirs, J_regressor, parents, lbs_weights must be consistent.
);
