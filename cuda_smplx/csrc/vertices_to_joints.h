#pragma once

#include <torch/extension.h>

torch::Tensor vertices_to_joints_cuda(
    torch::Tensor j_regressor,  // (num_joints, num_vertices) - typically sparse but passed as dense
    torch::Tensor v_shaped      // (batch_size, num_vertices, 3)
);

// Backward function declaration
std::vector<torch::Tensor> vertices_to_joints_backward_cuda(
    torch::Tensor grad_J_shaped,     // (batch_size, num_joints, 3)
    torch::Tensor j_regressor,       // (num_joints, num_vertices)
    torch::Tensor v_shaped           // (batch_size, num_vertices, 3)
    // Need j_regressor for dL/dv_shaped, and v_shaped for dL/dJ_regressor
);
