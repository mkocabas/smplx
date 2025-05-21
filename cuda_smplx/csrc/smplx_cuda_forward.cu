#include "smplx_cuda_forward.h"
#include "blend_shapes.h"
#include "vertices_to_joints.h"
#include "batch_rodrigues.h"
#include "batch_rigid_transform.h"
#include "skinning.h"

#include <torch/types.h> // For CUDAGuard, etc.

// Placeholder for number of joints. These should ideally be constants or derived.
// For SMPLX: Body 21 (excluding global), Jaw 1, Eyes 2, Hands 2*15 = 55 total joints in kinematic tree usually.
// However, the J_regressor might define more or less.
// Let's assume the 'num_joints' is consistent with parents, J_regressor, lbs_weights.
// num_body_joints for body_pose input, num_hand_joints for hand_pose input.

std::vector<torch::Tensor> smplx_forward_cuda(
    torch::Tensor betas,
    torch::Tensor expression, // specific to SMPLX
    torch::Tensor global_orient,
    torch::Tensor body_pose,
    torch::Tensor left_hand_pose,
    torch::Tensor right_hand_pose,
    torch::Tensor jaw_pose, // SMPLX

    torch::Tensor v_template,
    torch::Tensor shapedirs, // Should include shape AND expression blendshapes
    torch::Tensor posedirs,
    torch::Tensor J_regressor,
    torch::Tensor parents,
    torch::Tensor lbs_weights,

    bool use_pca_hands,
    torch::Tensor left_hand_components,
    torch::Tensor right_hand_components
) {
    // --- Input Checks (Comprehensive) ---
    TORCH_CHECK(betas.is_cuda(), "betas must be CUDA");
    TORCH_CHECK(expression.is_cuda(), "expression must be CUDA");
    TORCH_CHECK(global_orient.is_cuda(), "global_orient must be CUDA");
    TORCH_CHECK(body_pose.is_cuda(), "body_pose must be CUDA");
    TORCH_CHECK(left_hand_pose.is_cuda(), "left_hand_pose must be CUDA");
    TORCH_CHECK(right_hand_pose.is_cuda(), "right_hand_pose must be CUDA");
    TORCH_CHECK(jaw_pose.is_cuda(), "jaw_pose must be CUDA");
    
    TORCH_CHECK(betas.is_contiguous(), "betas must be contiguous");
    TORCH_CHECK(expression.is_contiguous(), "expression must be contiguous");
    TORCH_CHECK(global_orient.is_contiguous(), "global_orient must be contiguous");
    TORCH_CHECK(body_pose.is_contiguous(), "body_pose must be contiguous");
    // left_hand_pose & right_hand_pose contiguity checked after potential PCA
    TORCH_CHECK(jaw_pose.is_contiguous(), "jaw_pose must be contiguous");

    TORCH_CHECK(betas.scalar_type() == torch::kFloat32, "betas must be float32");
    TORCH_CHECK(expression.scalar_type() == torch::kFloat32, "expression must be float32");
    TORCH_CHECK(global_orient.scalar_type() == torch::kFloat32, "global_orient must be float32");
    TORCH_CHECK(body_pose.scalar_type() == torch::kFloat32, "body_pose must be float32");
    TORCH_CHECK(left_hand_pose.scalar_type() == torch::kFloat32, "left_hand_pose must be float32");
    TORCH_CHECK(right_hand_pose.scalar_type() == torch::kFloat32, "right_hand_pose must be float32");
    TORCH_CHECK(jaw_pose.scalar_type() == torch::kFloat32, "jaw_pose must be float32");

    TORCH_CHECK(shapedirs.is_cuda() && shapedirs.is_contiguous() && shapedirs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v_template.is_cuda() && v_template.is_contiguous() && v_template.scalar_type() == torch::kFloat32);
    TORCH_CHECK(J_regressor.is_cuda() && J_regressor.is_contiguous() && J_regressor.scalar_type() == torch::kFloat32);
    TORCH_CHECK(posedirs.is_cuda() && posedirs.is_contiguous() && posedirs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(lbs_weights.is_cuda() && lbs_weights.is_contiguous() && lbs_weights.scalar_type() == torch::kFloat32);
    // Parents tensor check is handled by batch_rigid_transform_cuda

    const int batch_size = betas.size(0);
    const int num_joints = parents.size(0); // Total joints in the kinematic model

    // --- 0. Prepare shape_components (betas + expression) ---
    // shapedirs is assumed to be (num_vertices, 3, num_betas + num_expression_coeffs)
    // The 'betas' input to blend_shapes_cuda should be the concatenation of shape betas and expression coeffs.
    torch::Tensor shape_components = torch::cat({betas, expression}, /*dim=*/1);
    TORCH_CHECK(shape_components.is_contiguous(), "shape_components (betas+expression) must be contiguous");
    TORCH_CHECK(shapedirs.size(2) == shape_components.size(1), 
        "shapedirs num_coeffs mismatch with concatenated betas+expression. Expected ", shapedirs.size(2), 
        " got ", shape_components.size(1));


    // --- 1. Blend Shapes (v_template + shape_components @ shapedirs) ---
    // v_shaped = v_template + blend_shapes(shape_components, shapedirs)
    // Output: v_shaped (batch_size, num_vertices, 3)
    torch::Tensor v_shaped = blend_shapes_cuda(shape_components, shapedirs, v_template);

    // --- 2. Vertices to Joints (J_regressor @ v_shaped) ---
    // J_shaped = J_regressor @ v_shaped
    // Output: J_shaped (batch_size, num_joints, 3)
    torch::Tensor J_shaped = vertices_to_joints_cuda(J_regressor, v_shaped);
    TORCH_CHECK(J_shaped.size(1) == num_joints, "J_shaped num_joints mismatch with parents tensor.");


    // --- 3. Prepare Full Pose & Convert to Rotation Matrices ---
    torch::Tensor actual_left_hand_pose = left_hand_pose;
    torch::Tensor actual_right_hand_pose = right_hand_pose;

    if (use_pca_hands) {
        TORCH_CHECK(left_hand_components.defined() && right_hand_components.defined(), "Hand PCA components needed");
        TORCH_CHECK(left_hand_components.is_cuda() && right_hand_components.is_cuda(), "Hand PCA components CUDA");
        TORCH_CHECK(left_hand_components.is_contiguous() && right_hand_components.is_contiguous(), "Hand PCA components must be contiguous");
        TORCH_CHECK(left_hand_components.scalar_type() == torch::kFloat32 && right_hand_components.scalar_type() == torch::kFloat32, "Hand PCA components must be float32");
        
        // Assuming left_hand_pose (B, n_pca) and left_hand_components (n_pca, n_feats_axis_angle)
        actual_left_hand_pose = torch::matmul(left_hand_pose, left_hand_components);
        actual_right_hand_pose = torch::matmul(right_hand_pose, right_hand_components);
    }
    // Ensure they are contiguous after potential matmul
    actual_left_hand_pose = actual_left_hand_pose.contiguous();
    actual_right_hand_pose = actual_right_hand_pose.contiguous();

    // Concatenate all pose parameters: global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose
    // The order must match the joint order in `parents`, `J_regressor`, `lbs_weights`.
    // This is a critical step and assumes a specific SMPL-X joint order.
    std::vector<torch::Tensor> pose_parts;
    pose_parts.push_back(global_orient.view({batch_size, 1, 3}));
    pose_parts.push_back(body_pose.view({batch_size, -1, 3})); // e.g., 21 joints for SMPLX body
    pose_parts.push_back(jaw_pose.view({batch_size, 1, 3}));
    // SMPLX typically has 15 joints for each hand.
    pose_parts.push_back(actual_left_hand_pose.view({batch_size, -1, 3})); 
    pose_parts.push_back(actual_right_hand_pose.view({batch_size, -1, 3}));
    // Add eye poses here if applicable and part of the kinematic chain for your model version

    torch::Tensor full_pose_axis_angle = torch::cat(pose_parts, /*dim=*/1); // Concatenate along the joint dimension

    int total_num_model_joints = full_pose_axis_angle.size(1);
    TORCH_CHECK(total_num_model_joints == num_joints, 
        "Concatenated pose parameters (", total_num_model_joints, 
        " joints) do not match number of joints in model definition (parents tensor: ", 
        num_joints, " joints).");

    torch::Tensor full_pose_flat = full_pose_axis_angle.reshape({batch_size * total_num_model_joints, 3});
    full_pose_flat = full_pose_flat.contiguous(); // Ensure contiguity after reshape

    // Output: rot_mats (B * total_num_model_joints, 3, 3)
    torch::Tensor rot_mats_flat = batch_rodrigues_cuda(full_pose_flat);

    // Reshape rot_mats to (B, total_num_model_joints, 3, 3) for subsequent steps
    torch::Tensor rot_mats = rot_mats_flat.view({batch_size, total_num_model_joints, 3, 3});
    rot_mats = rot_mats.contiguous();


    // --- 4. Batch Rigid Transform (Kinematic Chain) ---
    // Uses J_shaped (joints from current shape) and rot_mats (from current pose)
    // Output: posed_joints (B, J, 3), A_global (B, J, 4, 4)
    std::vector<torch::Tensor> transform_outputs = batch_rigid_transform_cuda(rot_mats, J_shaped, parents);
    torch::Tensor posed_joints = transform_outputs[0];
    torch::Tensor A_global = transform_outputs[1];

    // --- 5. Skinning ---
    // Output: final_vertices (B, V, 3)
    // skinning_cuda internally calculates pose_offsets using rot_mats and posedirs,
    // then adds to v_shaped to get v_posed, then applies LBS.
    torch::Tensor final_vertices = skinning_cuda(
        v_shaped,
        rot_mats, // rot_mats are used for pose_offsets calculation inside skinning_cuda
        posedirs,
        lbs_weights,
        A_global
    );

    return {final_vertices, posed_joints};
}
