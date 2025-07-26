#include <torch/extension.h>
#include <vector>

// CUDA function declarations
std::vector<torch::Tensor> batch_rigid_transform_cuda(
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents
);

torch::Tensor lbs_cuda(
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor transforms
);

torch::Tensor lbs_forward_cuda(
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents
);

// Backward pass
std::vector<torch::Tensor> lbs_backward_cuda(
    torch::Tensor grad_posed_vertices,
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor transforms
);

std::vector<torch::Tensor> batch_rigid_transform_backward_cuda(
    torch::Tensor grad_posed_joints,
    torch::Tensor grad_rel_transforms,
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents,
    torch::Tensor transform_chain
);


// Check tensor properties and move to CUDA if needed
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Python interface functions
std::vector<torch::Tensor> batch_rigid_transform(
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents
) {
    CHECK_INPUT(rot_mats);
    CHECK_INPUT(joints);
    CHECK_INPUT(parents);
    
    TORCH_CHECK(rot_mats.dim() == 4, "rot_mats must be 4D tensor [B, J, 3, 3]");
    TORCH_CHECK(joints.dim() == 3, "joints must be 3D tensor [B, J, 3]");
    TORCH_CHECK(parents.dim() == 1, "parents must be 1D tensor [J]");
    
    TORCH_CHECK(rot_mats.size(2) == 3 && rot_mats.size(3) == 3, 
                "rot_mats must have shape [..., 3, 3]");
    TORCH_CHECK(joints.size(2) == 3, "joints must have shape [..., 3]");
    
    TORCH_CHECK(rot_mats.size(0) == joints.size(0), 
                "rot_mats and joints must have same batch size");
    TORCH_CHECK(rot_mats.size(1) == joints.size(1), 
                "rot_mats and joints must have same number of joints");
    TORCH_CHECK(joints.size(1) == parents.size(0), 
                "joints and parents dimension mismatch");
    
    return batch_rigid_transform_cuda(rot_mats, joints, parents);
}

torch::Tensor lbs(
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor transforms
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(weights);
    CHECK_INPUT(transforms);
    
    TORCH_CHECK(vertices.dim() == 3, "vertices must be 3D tensor [B, V, 3]");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D tensor [V, J]");
    TORCH_CHECK(transforms.dim() == 4, "transforms must be 4D tensor [B, J, 4, 4]");
    
    TORCH_CHECK(vertices.size(2) == 3, "vertices must have shape [..., 3]");
    TORCH_CHECK(transforms.size(2) == 4 && transforms.size(3) == 4, 
                "transforms must have shape [..., 4, 4]");
    
    TORCH_CHECK(vertices.size(1) == weights.size(0), 
                "vertices and weights dimension mismatch");
    TORCH_CHECK(weights.size(1) == transforms.size(1), 
                "weights and transforms dimension mismatch");
    TORCH_CHECK(vertices.size(0) == transforms.size(0), 
                "vertices and transforms must have same batch size");
    
    return lbs_cuda(vertices, weights, transforms);
}

torch::Tensor lbs_forward(
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(weights);
    CHECK_INPUT(rot_mats);
    CHECK_INPUT(joints);
    CHECK_INPUT(parents);
    
    // Dimension checks
    TORCH_CHECK(vertices.dim() == 3, "vertices must be 3D tensor [B, V, 3]");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D tensor [V, J]");
    TORCH_CHECK(rot_mats.dim() == 4, "rot_mats must be 4D tensor [B, J, 3, 3]");
    TORCH_CHECK(joints.dim() == 3, "joints must be 3D tensor [B, J, 3]");
    TORCH_CHECK(parents.dim() == 1, "parents must be 1D tensor [J]");
    
    // Shape checks
    TORCH_CHECK(vertices.size(2) == 3, "vertices must have shape [..., 3]");
    TORCH_CHECK(rot_mats.size(2) == 3 && rot_mats.size(3) == 3, 
                "rot_mats must have shape [..., 3, 3]");
    TORCH_CHECK(joints.size(2) == 3, "joints must have shape [..., 3]");
    
    // Consistency checks
    TORCH_CHECK(vertices.size(0) == rot_mats.size(0) && 
                rot_mats.size(0) == joints.size(0), 
                "vertices, rot_mats, and joints must have same batch size");
    TORCH_CHECK(weights.size(0) == vertices.size(1), 
                "weights and vertices dimension mismatch");
    TORCH_CHECK(weights.size(1) == rot_mats.size(1) && 
                rot_mats.size(1) == joints.size(1) && 
                joints.size(1) == parents.size(0), 
                "joint dimensions must be consistent");
    
    return lbs_forward_cuda(vertices, weights, rot_mats, joints, parents);
}

// Binding for the backward functions
std::vector<torch::Tensor> lbs_backward(
    torch::Tensor grad_posed_vertices,
    torch::Tensor vertices,
    torch::Tensor weights,
    torch::Tensor transforms
) {
    CHECK_INPUT(grad_posed_vertices);
    CHECK_INPUT(vertices);
    CHECK_INPUT(weights);
    CHECK_INPUT(transforms);
    return lbs_backward_cuda(grad_posed_vertices, vertices, weights, transforms);
}

std::vector<torch::Tensor> batch_rigid_transform_backward(
    torch::Tensor grad_posed_joints,
    torch::Tensor grad_rel_transforms,
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents,
    torch::Tensor transform_chain
) {
    CHECK_INPUT(grad_posed_joints);
    CHECK_INPUT(grad_rel_transforms);
    CHECK_INPUT(rot_mats);
    CHECK_INPUT(joints);
    CHECK_INPUT(parents);
    CHECK_INPUT(transform_chain);
    return batch_rigid_transform_backward_cuda(
        grad_posed_joints, grad_rel_transforms, rot_mats, joints, parents, transform_chain
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_rigid_transform", &batch_rigid_transform, 
          "Batch rigid transformation (CUDA)");
    m.def("lbs", &lbs, 
          "Linear Blend Skinning (CUDA)");  
    m.def("lbs_forward", &lbs_forward, 
          "Combined LBS forward pass (CUDA)");
    m.def("lbs_backward", &lbs_backward,
          "LBS backward pass (CUDA)");
    m.def("batch_rigid_transform_backward", &batch_rigid_transform_backward,
            "Batch rigid transform backward pass (CUDA)");
}