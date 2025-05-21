#include "vertices_to_joints.h"
#include <torch/types.h> // Required for CUDAGuard

// CUDA kernel for vertices_to_joints
__global__ void vertices_to_joints_kernel(
    const float* __restrict__ j_regressor,
    const float* __restrict__ v_shaped,
    float* joints,
    int batch_size,
    int num_joints,
    int num_vertices,
    int num_dims // Should be 3 (x, y, z)
) {
    // j_regressor: (num_joints, num_vertices)
    // v_shaped: (batch_size, num_vertices, num_dims)
    // joints: (batch_size, num_joints, num_dims)

    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Joint index
    int d = blockIdx.z * blockDim.z + threadIdx.z; // Dimension index (x, y, z)

    if (b >= batch_size || j >= num_joints || d >= num_dims) {
        return;
    }

    float val = 0.0f;
    // Sum over num_vertices
    for (int v = 0; v < num_vertices; ++v) {
        // j_regressor[j, v] * v_shaped[b, v, d]
        val += j_regressor[j * num_vertices + v] * v_shaped[b * num_vertices * num_dims + v * num_dims + d];
    }
    
    joints[b * num_joints * num_dims + j * num_dims + d] = val;
}

torch::Tensor vertices_to_joints_cuda(
    torch::Tensor j_regressor,
    torch::Tensor v_shaped
) {
    TORCH_CHECK(j_regressor.is_cuda(), "j_regressor must be a CUDA tensor");
    TORCH_CHECK(v_shaped.is_cuda(), "v_shaped must be a CUDA tensor");

    TORCH_CHECK(j_regressor.is_contiguous(), "j_regressor must be contiguous");
    TORCH_CHECK(v_shaped.is_contiguous(), "v_shaped must be contiguous");

    TORCH_CHECK(j_regressor.scalar_type() == torch::kFloat32, "j_regressor must be a float32 tensor");
    TORCH_CHECK(v_shaped.scalar_type() == torch::kFloat32, "v_shaped must be a float32 tensor");

    TORCH_CHECK(j_regressor.dim() == 2, "j_regressor must be a 2D tensor");
    TORCH_CHECK(v_shaped.dim() == 3, "v_shaped must be a 3D tensor");

    const int num_joints_regressor = j_regressor.size(0);
    const int num_vertices_regressor = j_regressor.size(1);

    const int batch_size_vshaped = v_shaped.size(0);
    const int num_vertices_vshaped = v_shaped.size(1);
    const int num_dims_vshaped = v_shaped.size(2);

    TORCH_CHECK(num_vertices_regressor == num_vertices_vshaped, 
                "j_regressor and v_shaped must have the same number of vertices");
    TORCH_CHECK(num_dims_vshaped == 3, "v_shaped must have 3 dimensions (x,y,z)");

    const int batch_size = batch_size_vshaped;
    const int num_joints = num_joints_regressor;
    const int num_vertices = num_vertices_regressor;
    const int num_dims = num_dims_vshaped;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(v_shaped.device());
    torch::Tensor joints = torch::empty({batch_size, num_joints, num_dims}, options);

    const dim3 threads(8, 8, 3); // Example thread block size, adjust as needed
    const dim3 blocks(
        (batch_size + threads.x - 1) / threads.x,
        (num_joints + threads.y - 1) / threads.y,
        (num_dims + threads.z - 1) / threads.z
    );
    
    torch::cuda::CUDAGuard device_guard(v_shaped.device());

    vertices_to_joints_kernel<<<blocks, threads>>>(
        j_regressor.data_ptr<float>(),
        v_shaped.data_ptr<float>(),
        joints.data_ptr<float>(),
        batch_size,
        num_joints,
        num_vertices,
        num_dims
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed with error ", cudaGetErrorString(err));

    return joints;
}

// --- Backward Pass Kernels and Function ---

// Kernel for dL/dv_shaped
__global__ void vertices_to_joints_grad_v_shaped_kernel(
    const float* __restrict__ grad_J_shaped_data,    // (B, J, D) J=num_joints
    const float* __restrict__ j_regressor_data,      // (J, V) V=num_vertices
    float* grad_v_shaped_data,                       // (B, V, D)
    int batch_size,
    int num_joints,
    int num_vertices,
    int num_dims // Should be 3
) {
    // Grid: (batch_size, num_vertices, num_dims)
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int v = blockIdx.y * blockDim.y + threadIdx.y; // Vertex index
    int d = blockIdx.z * blockDim.z + threadIdx.z; // Dimension index

    if (b >= batch_size || v >= num_vertices || d >= num_dims) {
        return;
    }

    float sum_val = 0.0f;
    // Sum over num_joints (j)
    for (int j = 0; j < num_joints; ++j) {
        // grad_J_shaped[b,j,d] * j_regressor[j,v]
        sum_val += grad_J_shaped_data[b * num_joints * num_dims + j * num_dims + d] *
                   j_regressor_data[j * num_vertices + v];
    }
    grad_v_shaped_data[b * num_vertices * num_dims + v * num_dims + d] = sum_val;
}

// Kernel for dL/dJ_regressor
__global__ void vertices_to_joints_grad_j_regressor_kernel(
    const float* __restrict__ grad_J_shaped_data,    // (B, J, D)
    const float* __restrict__ v_shaped_data,         // (B, V, D)
    float* grad_j_regressor_data,                   // (J, V)
    int batch_size,
    int num_joints,
    int num_vertices,
    int num_dims
) {
    // Grid: (num_joints, num_vertices)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Joint index
    int v = blockIdx.y * blockDim.y + threadIdx.y; // Vertex index

    if (j >= num_joints || v >= num_vertices) {
        return;
    }

    float sum_val = 0.0f;
    // Sum over batch_size (b) and dimensions (d)
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < num_dims; ++d) {
            // grad_J_shaped[b,j,d] * v_shaped[b,v,d]
            sum_val += grad_J_shaped_data[b * num_joints * num_dims + j * num_dims + d] *
                       v_shaped_data[b * num_vertices * num_dims + v * num_dims + d];
        }
    }
    grad_j_regressor_data[j * num_vertices + v] = sum_val;
}


std::vector<torch::Tensor> vertices_to_joints_backward_cuda(
    torch::Tensor grad_J_shaped,
    torch::Tensor j_regressor,
    torch::Tensor v_shaped
) {
    TORCH_CHECK(grad_J_shaped.is_cuda() && j_regressor.is_cuda() && v_shaped.is_cuda());
    TORCH_CHECK(grad_J_shaped.is_contiguous() && j_regressor.is_contiguous() && v_shaped.is_contiguous());
    TORCH_CHECK(grad_J_shaped.scalar_type() == torch::kFloat32, "grad_J_shaped must be float32");
    TORCH_CHECK(j_regressor.scalar_type() == torch::kFloat32, "j_regressor must be float32");
    TORCH_CHECK(v_shaped.scalar_type() == torch::kFloat32, "v_shaped must be float32");


    const int batch_size = grad_J_shaped.size(0);
    const int num_joints = grad_J_shaped.size(1);
    const int num_dims = grad_J_shaped.size(2); // Should be 3
    TORCH_CHECK(num_dims == 3, "Number of dimensions must be 3 for vertices_to_joints_backward_cuda.");


    const int num_vertices = v_shaped.size(1);
    TORCH_CHECK(j_regressor.size(0) == num_joints && j_regressor.size(1) == num_vertices, "j_regressor dimension mismatch");
    TORCH_CHECK(v_shaped.size(0) == batch_size && v_shaped.size(2) == num_dims, "v_shaped dimension mismatch");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_J_shaped.device());
    torch::Tensor grad_v_shaped = torch::empty_like(v_shaped, options);
    torch::Tensor grad_j_regressor = torch::empty_like(j_regressor, options);

    torch::cuda::CUDAGuard device_guard(grad_J_shaped.device());
    cudaError_t err;

    // --- Calculate grad_v_shaped ---
    dim3 threads_gvs(8, 8, 3); // Example for B, V, D
    dim3 blocks_gvs( (batch_size + threads_gvs.x - 1) / threads_gvs.x,
                     (num_vertices + threads_gvs.y - 1) / threads_gvs.y,
                     (num_dims + threads_gvs.z - 1) / threads_gvs.z );
    vertices_to_joints_grad_v_shaped_kernel<<<blocks_gvs, threads_gvs>>>(
        grad_J_shaped.data_ptr<float>(), j_regressor.data_ptr<float>(), grad_v_shaped.data_ptr<float>(),
        batch_size, num_joints, num_vertices, num_dims
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "vertices_to_joints_grad_v_shaped_kernel launch failed: ", cudaGetErrorString(err));

    // --- Calculate grad_j_regressor ---
    // (J_regressor is often fixed, but included for completeness)
    dim3 threads_gjr(16, 16); // Example for J, V
    dim3 blocks_gjr( (num_joints + threads_gjr.x - 1) / threads_gjr.x,
                     (num_vertices + threads_gjr.y - 1) / threads_gjr.y );
    vertices_to_joints_grad_j_regressor_kernel<<<blocks_gjr, threads_gjr>>>(
        grad_J_shaped.data_ptr<float>(), v_shaped.data_ptr<float>(), grad_j_regressor.data_ptr<float>(),
        batch_size, num_joints, num_vertices, num_dims
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "vertices_to_joints_grad_j_regressor_kernel launch failed: ", cudaGetErrorString(err));

    // Return gradients: dJ_regressor, dv_shaped
    // Order must match inputs of forward: j_regressor, v_shaped
    return {grad_j_regressor, grad_v_shaped};
}
