#include "blend_shapes.h"
#include <torch/types.h> // Required for CUDAGuard

// CUDA kernel for blend_shapes
__global__ void blend_shapes_kernel(
    const float* __restrict__ betas,
    const float* __restrict__ shapedirs,
    const float* __restrict__ v_template,
    float* v_shaped,
    int batch_size,
    int num_vertices,
    int num_betas,
    int num_dims // Should be 3 (x, y, z)
) {
    // betas: (batch_size, num_betas)
    // shapedirs: (num_vertices, num_dims, num_betas)
    // v_template: (num_vertices, num_dims)
    // v_shaped: (batch_size, num_vertices, num_dims)

    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int v = blockIdx.y * blockDim.y + threadIdx.y; // Vertex index
    int d = blockIdx.z * blockDim.z + threadIdx.z; // Dimension index (x, y, z)

    if (b >= batch_size || v >= num_vertices || d >= num_dims) {
        return;
    }

    float val = 0.0f;
    // Sum over num_betas
    for (int l = 0; l < num_betas; ++l) {
        // betas[b, l] * shapedirs[v, d, l]
        val += betas[b * num_betas + l] * shapedirs[v * num_dims * num_betas + d * num_betas + l];
    }
    
    // Add v_template contribution
    // v_shaped[b, v, d] = v_template[v, d] + val;
    v_shaped[b * num_vertices * num_dims + v * num_dims + d] = v_template[v * num_dims + d] + val;
}

torch::Tensor blend_shapes_cuda(
    torch::Tensor betas,
    torch::Tensor shapedirs,
    torch::Tensor v_template
) {
    TORCH_CHECK(betas.is_cuda(), "betas must be a CUDA tensor");
    TORCH_CHECK(shapedirs.is_cuda(), "shapedirs must be a CUDA tensor");
    TORCH_CHECK(v_template.is_cuda(), "v_template must be a CUDA tensor");

    TORCH_CHECK(betas.is_contiguous(), "betas must be contiguous");
    TORCH_CHECK(shapedirs.is_contiguous(), "shapedirs must be contiguous");
    TORCH_CHECK(v_template.is_contiguous(), "v_template must be contiguous");
    
    TORCH_CHECK(betas.scalar_type() == torch::kFloat32, "betas must be a float32 tensor");
    TORCH_CHECK(shapedirs.scalar_type() == torch::kFloat32, "shapedirs must be a float32 tensor");
    TORCH_CHECK(v_template.scalar_type() == torch::kFloat32, "v_template must be a float32 tensor");

    const int batch_size = betas.size(0);
    const int num_betas_betas = betas.size(1);

    const int num_vertices_shapedirs = shapedirs.size(0);
    const int num_dims_shapedirs = shapedirs.size(1);
    const int num_betas_shapedirs = shapedirs.size(2);

    const int num_vertices_template = v_template.size(0);
    const int num_dims_template = v_template.size(1);

    TORCH_CHECK(num_dims_shapedirs == num_dims_template, "shapedirs and v_template must have the same number of dimensions (3)");
    TORCH_CHECK(num_betas_betas == num_betas_shapedirs, "betas and shapedirs must have the same number of betas");
    TORCH_CHECK(num_vertices_shapedirs == num_vertices_template, "shapedirs and v_template must have the same number of vertices");
    TORCH_CHECK(num_dims_shapedirs == 3, "Number of dimensions for shapedirs must be 3.");
    
    const int num_dims = num_dims_shapedirs;
    const int num_vertices = num_vertices_shapedirs;
    const int num_betas = num_betas_betas;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(betas.device());
    torch::Tensor v_shaped = torch::empty({batch_size, num_vertices, num_dims}, options);

    const dim3 threads(16, 16, 1); // Example thread block size
    const dim3 blocks(
        (batch_size + threads.x - 1) / threads.x,
        (num_vertices + threads.y - 1) / threads.y,
        (num_dims + threads.z - 1) / threads.z
    );
    
    // Ensure that device is set properly for the kernel launch
    torch::cuda::CUDAGuard device_guard(betas.device());

    blend_shapes_kernel<<<blocks, threads>>>(
        betas.data_ptr<float>(),
        shapedirs.data_ptr<float>(),
        v_template.data_ptr<float>(),
        v_shaped.data_ptr<float>(),
        batch_size,
        num_vertices,
        num_betas,
        num_dims
    );
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed with error ", cudaGetErrorString(err));

    return v_shaped;
}

// --- Backward Pass Kernels and Function ---

// Kernel for dL/dbetas
__global__ void blend_shapes_grad_betas_kernel(
    const float* __restrict__ grad_v_shaped_data, // (B, V, D)
    const float* __restrict__ shapedirs_data,     // (V, D, L) L = num_betas
    float* grad_betas_data,                       // (B, L)
    int batch_size,
    int num_vertices,
    int num_dims, // Should be 3
    int num_betas 
) {
    // Grid: (batch_size, num_betas)
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int l = blockIdx.y * blockDim.y + threadIdx.y; // Beta index (L)

    if (b >= batch_size || l >= num_betas) {
        return;
    }

    float sum_val = 0.0f;
    // Sum over vertices (v) and dimensions (d)
    for (int v = 0; v < num_vertices; ++v) {
        for (int d = 0; d < num_dims; ++d) {
            // grad_v_shaped[b,v,d] * shapedirs[v,d,l]
            sum_val += grad_v_shaped_data[b * num_vertices * num_dims + v * num_dims + d] *
                       shapedirs_data[v * num_dims * num_betas + d * num_betas + l];
        }
    }
    grad_betas_data[b * num_betas + l] = sum_val;
}

// Kernel for dL/dshapedirs
__global__ void blend_shapes_grad_shapedirs_kernel(
    const float* __restrict__ grad_v_shaped_data, // (B, V, D)
    const float* __restrict__ betas_data,         // (B, L)
    float* grad_shapedirs_data,                   // (V, D, L)
    int batch_size,
    int num_vertices,
    int num_dims,
    int num_betas
) {
    // Grid: (num_vertices, num_dims, num_betas)
    int v = blockIdx.x * blockDim.x + threadIdx.x; // Vertex index
    int d = blockIdx.y * blockDim.y + threadIdx.y; // Dimension index
    int l = blockIdx.z * blockDim.z + threadIdx.z; // Beta index

    if (v >= num_vertices || d >= num_dims || l >= num_betas) {
        return;
    }

    float sum_val = 0.0f;
    // Sum over batch_size (b)
    for (int b = 0; b < batch_size; ++b) {
        // grad_v_shaped[b,v,d] * betas[b,l]
        sum_val += grad_v_shaped_data[b * num_vertices * num_dims + v * num_dims + d] *
                   betas_data[b * num_betas + l];
    }
    grad_shapedirs_data[v * num_dims * num_betas + d * num_betas + l] = sum_val;
}

// Kernel for dL/dv_template
__global__ void blend_shapes_grad_v_template_kernel(
    const float* __restrict__ grad_v_shaped_data, // (B, V, D)
    float* grad_v_template_data,                  // (V, D)
    int batch_size,
    int num_vertices,
    int num_dims
) {
    // Grid: (num_vertices, num_dims)
    int v = blockIdx.x * blockDim.x + threadIdx.x; // Vertex index
    int d = blockIdx.y * blockDim.y + threadIdx.y; // Dimension index

    if (v >= num_vertices || d >= num_dims) {
        return;
    }

    float sum_val = 0.0f;
    // Sum over batch_size (b)
    for (int b = 0; b < batch_size; ++b) {
        sum_val += grad_v_shaped_data[b * num_vertices * num_dims + v * num_dims + d];
    }
    grad_v_template_data[v * num_dims + d] = sum_val;
}


std::vector<torch::Tensor> blend_shapes_backward_cuda(
    torch::Tensor grad_v_shaped,
    torch::Tensor betas,
    torch::Tensor shapedirs,
    torch::Tensor v_template // v_template isn't used for grad_betas/grad_shapedirs but passed for consistency
) {
    TORCH_CHECK(grad_v_shaped.is_cuda() && betas.is_cuda() && shapedirs.is_cuda() && v_template.is_cuda());
    TORCH_CHECK(grad_v_shaped.is_contiguous() && betas.is_contiguous() && shapedirs.is_contiguous() && v_template.is_contiguous());
    // Add dtype checks...
    TORCH_CHECK(grad_v_shaped.scalar_type() == torch::kFloat32, "grad_v_shaped must be float32");
    TORCH_CHECK(betas.scalar_type() == torch::kFloat32, "betas must be float32");
    TORCH_CHECK(shapedirs.scalar_type() == torch::kFloat32, "shapedirs must be float32");
    TORCH_CHECK(v_template.scalar_type() == torch::kFloat32, "v_template must be float32");


    const int batch_size = grad_v_shaped.size(0);
    const int num_vertices = grad_v_shaped.size(1);
    const int num_dims = grad_v_shaped.size(2); // Should be 3

    const int num_betas_betas = betas.size(1);
    const int num_betas_shapedirs = shapedirs.size(2);
    TORCH_CHECK(num_betas_betas == num_betas_shapedirs, "num_betas mismatch between betas and shapedirs");
    const int num_betas = num_betas_betas;

    TORCH_CHECK(shapedirs.size(0) == num_vertices && shapedirs.size(1) == num_dims, "shapedirs dimension mismatch");
    TORCH_CHECK(v_template.size(0) == num_vertices && v_template.size(1) == num_dims, "v_template dimension mismatch");
    TORCH_CHECK(num_dims == 3, "Number of dimensions must be 3.");


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_v_shaped.device());
    torch::Tensor grad_betas = torch::empty_like(betas, options); // (B, L)
    torch::Tensor grad_shapedirs = torch::empty_like(shapedirs, options); // (V, D, L)
    torch::Tensor grad_v_template = torch::empty_like(v_template, options); // (V, D)

    torch::cuda::CUDAGuard device_guard(grad_v_shaped.device());
    cudaError_t err;

    // --- Calculate grad_betas ---
    dim3 threads_gb(16, 16); // Example
    dim3 blocks_gb((batch_size + threads_gb.x - 1) / threads_gb.x,
                   (num_betas + threads_gb.y - 1) / threads_gb.y);
    blend_shapes_grad_betas_kernel<<<blocks_gb, threads_gb>>>(
        grad_v_shaped.data_ptr<float>(), shapedirs.data_ptr<float>(), grad_betas.data_ptr<float>(),
        batch_size, num_vertices, num_dims, num_betas
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "blend_shapes_grad_betas_kernel launch failed: ", cudaGetErrorString(err));

    // --- Calculate grad_shapedirs ---
    dim3 threads_gs(8, 2, 8); // Example for V, D, L
    // Ensure num_dims is handled correctly for threads_gs.y (max 3 for D)
    // If num_dims is always 3, threads_gs.y could be 3. If it can be less, use min(num_dims, 2) or adjust.
    // For simplicity, assuming num_dims is 3, so threads_gs.y=2 is okay, or 3.
    // If num_dims=1, threads_gs.y=1. If num_dims=2, threads_gs.y=2. If num_dims=3, threads_gs.y could be 1,2, or 3.
    // Let's make it more robust:
    dim3 actual_threads_gs = threads_gs;
    if (num_dims < threads_gs.y) actual_threads_gs.y = num_dims;

    dim3 blocks_gs((num_vertices + actual_threads_gs.x - 1) / actual_threads_gs.x,
                   (num_dims + actual_threads_gs.y - 1) / actual_threads_gs.y,
                   (num_betas + actual_threads_gs.z - 1) / actual_threads_gs.z);
    blend_shapes_grad_shapedirs_kernel<<<blocks_gs, actual_threads_gs>>>(
        grad_v_shaped.data_ptr<float>(), betas.data_ptr<float>(), grad_shapedirs.data_ptr<float>(),
        batch_size, num_vertices, num_dims, num_betas
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "blend_shapes_grad_shapedirs_kernel launch failed: ", cudaGetErrorString(err));

    // --- Calculate grad_v_template ---
    dim3 threads_gvt(16, 3); // Example for V, D
    // Ensure num_dims is handled correctly for threads_gvt.y
    dim3 actual_threads_gvt = threads_gvt;
    if (num_dims < threads_gvt.y) actual_threads_gvt.y = num_dims;
    
    dim3 blocks_gvt((num_vertices + actual_threads_gvt.x - 1) / actual_threads_gvt.x,
                    (num_dims + actual_threads_gvt.y - 1) / actual_threads_gvt.y);
    blend_shapes_grad_v_template_kernel<<<blocks_gvt, actual_threads_gvt>>>(
        grad_v_shaped.data_ptr<float>(), grad_v_template.data_ptr<float>(),
        batch_size, num_vertices, num_dims
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "blend_shapes_grad_v_template_kernel launch failed: ", cudaGetErrorString(err));

    // Return gradients: d_betas, d_shapedirs, d_v_template
    // Order should match inputs of forward function for autograd: betas, shapedirs, v_template
    return {grad_betas, grad_shapedirs, grad_v_template};
}
