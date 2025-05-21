#pragma once

#include <torch/extension.h>

torch::Tensor batch_rodrigues_cuda(
    torch::Tensor rot_vecs // (N, 3) where N = batch_size * num_total_joints
);

// Backward function declaration
torch::Tensor batch_rodrigues_backward_cuda(
    torch::Tensor grad_rot_mats, // (N, 3, 3)
    torch::Tensor rot_vecs,      // (N, 3) - original input to forward
    torch::Tensor rot_mats_fwd   // (N, 3, 3) - output of forward (C)
);
