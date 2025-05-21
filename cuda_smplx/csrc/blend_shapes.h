#pragma once

#include <torch/extension.h>

torch::Tensor blend_shapes_cuda(
    torch::Tensor betas,         // (batch_size, num_betas)
    torch::Tensor shapedirs,     // (num_vertices, 3, num_betas)
    torch::Tensor v_template     // (num_vertices, 3)
);

// Backward function declaration
std::vector<torch::Tensor> blend_shapes_backward_cuda(
    torch::Tensor grad_v_shaped,     // (batch_size, num_vertices, 3)
    torch::Tensor betas,             // (batch_size, num_betas) - num_betas here is num_shape_coeffs + num_expr_coeffs
    torch::Tensor shapedirs,         // (num_vertices, 3, num_betas)
    torch::Tensor v_template         // (num_vertices, 3)
    // We need betas and shapedirs to compute gradients for each other.
    // v_template is not strictly needed for grad_betas or grad_shapedirs, but for grad_v_template.
);
