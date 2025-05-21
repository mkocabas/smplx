#include "smplx_cuda_backward.h"
#include "skinning.h"          // For skinning_backward_cuda
#include "batch_rigid_transform.h" // For batch_rigid_transform_backward_cuda
#include "batch_rodrigues.h"   // For batch_rodrigues_backward_cuda
#include "vertices_to_joints.h"// For vertices_to_joints_backward_cuda
#include "blend_shapes.h"      // For blend_shapes_backward_cuda

#include <torch/types.h>
#include <vector>


std::vector<torch::Tensor> smplx_backward_cuda(
    torch::Tensor grad_final_vertices,
    torch::Tensor grad_posed_joints, 

    torch::Tensor betas_fwd,
    torch::Tensor expression_fwd,
    torch::Tensor global_orient_fwd,
    torch::Tensor body_pose_fwd,
    torch::Tensor left_hand_pose_fwd,
    torch::Tensor right_hand_pose_fwd,
    torch::Tensor jaw_pose_fwd,
    
    torch::Tensor v_template_fwd,
    torch::Tensor shapedirs_fwd, 
    torch::Tensor posedirs_fwd,
    torch::Tensor J_regressor_fwd,
    torch::Tensor parents_fwd,
    torch::Tensor lbs_weights_fwd,
    
    bool use_pca_hands_fwd,
    torch::Tensor left_hand_components_fwd,
    torch::Tensor right_hand_components_fwd,

    torch::Tensor v_shaped_fwd,
    torch::Tensor J_shaped_fwd,
    torch::Tensor full_pose_axis_angle_fwd, 
    torch::Tensor rot_mats_fwd,
    torch::Tensor posed_joints_fwd, 
    torch::Tensor A_global_fwd,
    torch::Tensor v_posed_fwd
) {
    // --- Input Checks ---
    TORCH_CHECK(grad_final_vertices.is_cuda() && grad_final_vertices.is_contiguous() && grad_final_vertices.scalar_type() == torch::kFloat32);
    TORCH_CHECK(grad_posed_joints.is_cuda() && grad_posed_joints.is_contiguous() && grad_posed_joints.scalar_type() == torch::kFloat32);

    TORCH_CHECK(betas_fwd.is_cuda() && betas_fwd.is_contiguous() && betas_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(expression_fwd.is_cuda() && expression_fwd.is_contiguous() && expression_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(global_orient_fwd.is_cuda() && global_orient_fwd.is_contiguous() && global_orient_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(body_pose_fwd.is_cuda() && body_pose_fwd.is_contiguous() && body_pose_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(left_hand_pose_fwd.is_cuda() && left_hand_pose_fwd.is_contiguous() && left_hand_pose_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(right_hand_pose_fwd.is_cuda() && right_hand_pose_fwd.is_contiguous() && right_hand_pose_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(jaw_pose_fwd.is_cuda() && jaw_pose_fwd.is_contiguous() && jaw_pose_fwd.scalar_type() == torch::kFloat32);

    TORCH_CHECK(v_template_fwd.is_cuda() && v_template_fwd.is_contiguous() && v_template_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(shapedirs_fwd.is_cuda() && shapedirs_fwd.is_contiguous() && shapedirs_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(posedirs_fwd.is_cuda() && posedirs_fwd.is_contiguous() && posedirs_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(J_regressor_fwd.is_cuda() && J_regressor_fwd.is_contiguous() && J_regressor_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(parents_fwd.is_contiguous()); // Device check handled by individual backward calls
    TORCH_CHECK(lbs_weights_fwd.is_cuda() && lbs_weights_fwd.is_contiguous() && lbs_weights_fwd.scalar_type() == torch::kFloat32);

    if (use_pca_hands_fwd) {
        TORCH_CHECK(left_hand_components_fwd.defined() && left_hand_components_fwd.numel() > 0 && left_hand_components_fwd.is_cuda() && left_hand_components_fwd.is_contiguous() && left_hand_components_fwd.scalar_type() == torch::kFloat32);
        TORCH_CHECK(right_hand_components_fwd.defined() && right_hand_components_fwd.numel() > 0 && right_hand_components_fwd.is_cuda() && right_hand_components_fwd.is_contiguous() && right_hand_components_fwd.scalar_type() == torch::kFloat32);
    }

    TORCH_CHECK(v_shaped_fwd.is_cuda() && v_shaped_fwd.is_contiguous() && v_shaped_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(J_shaped_fwd.is_cuda() && J_shaped_fwd.is_contiguous() && J_shaped_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(full_pose_axis_angle_fwd.is_cuda() && full_pose_axis_angle_fwd.is_contiguous() && full_pose_axis_angle_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rot_mats_fwd.is_cuda() && rot_mats_fwd.is_contiguous() && rot_mats_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(posed_joints_fwd.is_cuda() && posed_joints_fwd.is_contiguous() && posed_joints_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(A_global_fwd.is_cuda() && A_global_fwd.is_contiguous() && A_global_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v_posed_fwd.is_cuda() && v_posed_fwd.is_contiguous() && v_posed_fwd.scalar_type() == torch::kFloat32);


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_final_vertices.device());
    torch::Tensor grad_rot_mats_accum = torch::zeros_like(rot_mats_fwd, options);

    // 5. Backward for skinning_cuda
    std::vector<torch::Tensor> skinning_grads = skinning_backward_cuda(
        grad_final_vertices, v_shaped_fwd, rot_mats_fwd, posedirs_fwd,
        lbs_weights_fwd, A_global_fwd, v_posed_fwd
    );
    torch::Tensor grad_v_shaped_from_skinning = skinning_grads[0];
    torch::Tensor grad_rot_mats_from_skinning_po = skinning_grads[1];
    torch::Tensor grad_posedirs = skinning_grads[2];
    torch::Tensor grad_lbs_weights = skinning_grads[3];
    torch::Tensor grad_A_global_from_skinning = skinning_grads[4];
    
    grad_rot_mats_accum.add_(grad_rot_mats_from_skinning_po);

    // 4. Backward for batch_rigid_transform_cuda
    std::vector<torch::Tensor> brt_grads = batch_rigid_transform_backward_cuda(
        grad_posed_joints, grad_A_global_from_skinning, rot_mats_fwd,
        J_shaped_fwd, parents_fwd, posed_joints_fwd, A_global_fwd
    );
    torch::Tensor grad_rot_mats_from_brt = brt_grads[0];
    torch::Tensor grad_J_shaped_from_brt = brt_grads[1];
    
    grad_rot_mats_accum.add_(grad_rot_mats_from_brt);
    
    // 3. Backward for batch_rodrigues_cuda
    const int B = rot_mats_fwd.size(0);
    const int J = rot_mats_fwd.size(1); // Total number of joints in the kinematic chain
    torch::Tensor grad_rot_mats_accum_flat = grad_rot_mats_accum.reshape({B * J, 3, 3});
    
    torch::Tensor grad_full_pose_flat = batch_rodrigues_backward_cuda(
        grad_rot_mats_accum_flat,
        full_pose_axis_angle_fwd.reshape({B * J, 3}),
        rot_mats_fwd.reshape({B * J, 3, 3})
    );
    torch::Tensor grad_full_pose_axis_angle = grad_full_pose_flat.reshape({B, J, 3});

    // Define joint counts for slicing (assuming SMPL-X structure)
    const int NUM_GLOBAL_JOINTS = 1;
    const int NUM_BODY_JOINTS = 21; // Example: SMPL body joints
    const int NUM_JAW_JOINTS = 1;   // Example: SMPL-X jaw joint
    const int NUM_HAND_JOINTS = 15; // Example: Per hand
    // Eye joints are not included in this example based on forward pass inputs

    torch::Tensor grad_global_orient, grad_body_pose, grad_jaw_pose, grad_left_hand_pose, grad_right_hand_pose;
    int current_idx = 0;

    grad_global_orient = grad_full_pose_axis_angle.slice(/*dim=*/1, current_idx, current_idx + NUM_GLOBAL_JOINTS).reshape({B, -1});
    current_idx += NUM_GLOBAL_JOINTS;
    grad_body_pose = grad_full_pose_axis_angle.slice(/*dim=*/1, current_idx, current_idx + NUM_BODY_JOINTS).reshape({B, -1});
    current_idx += NUM_BODY_JOINTS;
    grad_jaw_pose = grad_full_pose_axis_angle.slice(/*dim=*/1, current_idx, current_idx + NUM_JAW_JOINTS).reshape({B, -1});
    current_idx += NUM_JAW_JOINTS;
    
    torch::Tensor grad_actual_left_hand_aa = grad_full_pose_axis_angle.slice(/*dim=*/1, current_idx, current_idx + NUM_HAND_JOINTS).reshape({B, -1});
    current_idx += NUM_HAND_JOINTS;
    torch::Tensor grad_actual_right_hand_aa = grad_full_pose_axis_angle.slice(/*dim=*/1, current_idx, current_idx + NUM_HAND_JOINTS).reshape({B, -1});
    // current_idx += NUM_HAND_JOINTS; // If there were more parts after this

    if (use_pca_hands_fwd) {
        grad_left_hand_pose = torch::matmul(grad_actual_left_hand_aa, left_hand_components_fwd.transpose(0,1));
        grad_right_hand_pose = torch::matmul(grad_actual_right_hand_aa, right_hand_components_fwd.transpose(0,1));
    } else {
        grad_left_hand_pose = grad_actual_left_hand_aa;
        grad_right_hand_pose = grad_actual_right_hand_aa;
    }
    
    // 2. Backward for vertices_to_joints_cuda
    std::vector<torch::Tensor> v2j_grads = vertices_to_joints_backward_cuda(
        grad_J_shaped_from_brt, J_regressor_fwd, v_shaped_fwd
    );
    torch::Tensor grad_J_regressor = v2j_grads[0];
    torch::Tensor grad_v_shaped_from_v2j = v2j_grads[1];

    // 1. Backward for blend_shapes_cuda
    torch::Tensor total_grad_v_shaped = grad_v_shaped_from_skinning.add(grad_v_shaped_from_v2j);
    
    torch::Tensor shape_components_fwd = torch::cat({betas_fwd, expression_fwd}, 1).contiguous();

    std::vector<torch::Tensor> bs_grads = blend_shapes_backward_cuda(
        total_grad_v_shaped, shape_components_fwd, shapedirs_fwd, v_template_fwd
    );
    torch::Tensor grad_shape_components = bs_grads[0];
    torch::Tensor grad_shapedirs = bs_grads[1]; 
    torch::Tensor grad_v_template = bs_grads[2];

    int num_betas = betas_fwd.size(1);
    torch::Tensor grad_betas = grad_shape_components.slice(/*dim=*/1, 0, num_betas);
    torch::Tensor grad_expression = grad_shape_components.slice(/*dim=*/1, num_betas, grad_shape_components.size(1));

    return {
        grad_betas, grad_expression,
        grad_global_orient, grad_body_pose, grad_left_hand_pose, grad_right_hand_pose, grad_jaw_pose,
        grad_v_template, grad_shapedirs, grad_posedirs, grad_J_regressor,
        torch::Tensor(), // grad_parents (None)
        grad_lbs_weights,
        torch::Tensor(), // grad_use_pca_hands (None)
        torch::Tensor(), // grad_left_hand_components (None)
        torch::Tensor()  // grad_right_hand_components (None)
    };
}
