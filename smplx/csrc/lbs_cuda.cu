#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Helper function for 4x4 matrix multiplication on device
__device__ void matmul_4x4(const float* A, const float* B, float* C) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += A[i * 4 + k] * B[k * 4 + j];
            }
            C[i * 4 + j] = sum;
        }
    }
}

// CUDA kernel for full batch rigid transform with kinematic chain
__global__ void batch_rigid_transform_kernel(
    const float* __restrict__ rot_mats,      // [B, J, 3, 3]
    const float* __restrict__ joints,        // [B, J, 3]
    const float* __restrict__ parents,       // [J] (int parents, but passed as float)
    float* __restrict__ posed_joints,        // [B, J, 3] output
    float* __restrict__ rel_transforms,      // [B, J, 4, 4] output
    int B, int J
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    // Each batch is processed by one block
    extern __shared__ float shared_mem[];
    
    // Shared memory layout:
    // transforms_mat: J * 16 floats (local transforms for each joint)
    // transform_chain: J * 16 floats (accumulated transforms)
    float* transforms_mat = shared_mem;
    float* transform_chain = shared_mem + J * 16;
    
    int tid = threadIdx.x;
    
    // Step 1: Compute relative joints and local transformation matrices
    for (int j = tid; j < J; j += blockDim.x) {
        int joint_base = batch_idx * J * 3 + j * 3;
        int rot_base = batch_idx * J * 9 + j * 9;
        int transform_base = j * 16;
        
        // Get joint position
        float joint_pos[3] = {joints[joint_base], joints[joint_base + 1], joints[joint_base + 2]};
        
        // Compute relative joint position
        float rel_joint[3];
        if (j == 0) {
            rel_joint[0] = joint_pos[0];
            rel_joint[1] = joint_pos[1]; 
            rel_joint[2] = joint_pos[2];
        } else {
            int parent_idx = (int)parents[j];
            int parent_base = batch_idx * J * 3 + parent_idx * 3;
            rel_joint[0] = joint_pos[0] - joints[parent_base];
            rel_joint[1] = joint_pos[1] - joints[parent_base + 1];
            rel_joint[2] = joint_pos[2] - joints[parent_base + 2];
        }
        
        // Create 4x4 transformation matrix: [R | t; 0 0 0 1]
        // Copy rotation matrix (3x3)
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                transforms_mat[transform_base + i * 4 + k] = rot_mats[rot_base + i * 3 + k];
            }
            // Set translation
            transforms_mat[transform_base + i * 4 + 3] = rel_joint[i];
        }
        // Set bottom row [0, 0, 0, 1]
        transforms_mat[transform_base + 12] = 0.0f;
        transforms_mat[transform_base + 13] = 0.0f;
        transforms_mat[transform_base + 14] = 0.0f;
        transforms_mat[transform_base + 15] = 1.0f;
    }
    
    __syncthreads();
    
    // Step 2: Compute kinematic chain (sequential, but parallelized per batch)
    if (tid == 0) {
        // Initialize root transform
        for (int i = 0; i < 16; i++) {
            transform_chain[i] = transforms_mat[i];
        }
        
        // Process joints in order (this must be sequential)
        for (int j = 1; j < J; j++) {
            int parent_idx = (int)parents[j];
            
            // Multiply: transform_chain[parent_idx] * transforms_mat[j]
            matmul_4x4(&transform_chain[parent_idx * 16], 
                      &transforms_mat[j * 16], 
                      &transform_chain[j * 16]);
        }
    }
    
    __syncthreads();
    
    // Step 3: Extract posed joints and compute relative transforms
    for (int j = tid; j < J; j += blockDim.x) {
        int joint_base = batch_idx * J * 3 + j * 3;
        int transform_base = batch_idx * J * 16 + j * 16;
        
        // Extract posed joint positions (last column of transform matrix)
        posed_joints[joint_base]     = transform_chain[j * 16 + 3];
        posed_joints[joint_base + 1] = transform_chain[j * 16 + 7];
        posed_joints[joint_base + 2] = transform_chain[j * 16 + 11];
        
        // Compute relative transforms: transforms - F.pad(transforms * joints_homogen, [3, 0, 0, 0, 0, 0, 0, 0])
        // The joint_homogen uses the unsqueezed original joint positions
        // In CPU: joints_homogen = F.pad(joints_unsqueezed, [0, 0, 0, 1]) where joints_unsqueezed is [B,J,3,1]
        // F.pad([0, 0, 0, 1]) adds a row of zeros, so joints_homogen is [B,J,4,1] with 0 in the 4th element
        float joint_homogen[4] = {joints[joint_base], joints[joint_base + 1], joints[joint_base + 2], 0.0f};
        
        // Compute transform * joint_homogen
        float transformed_joint[4];
        for (int i = 0; i < 4; i++) {
            transformed_joint[i] = 0.0f;
            for (int k = 0; k < 4; k++) {
                transformed_joint[i] += transform_chain[j * 16 + i * 4 + k] * joint_homogen[k];
            }
        }
        
        // Compute relative transform: T - [0, 0, 0, transformed_joint]
        // F.pad([transformed_joint], [3, 0, 0, 0, 0, 0, 0, 0]) puts transformed_joint in the last column only
        // BUT the joints_homogen tensor in CPU has 0 as the last element, not 1!
        // So we need to recreate that behavior
        for (int i = 0; i < 16; i++) {
            int row = i / 4;
            int col = i % 4;
            float subtract_val = 0.0f;
            
            if (col == 3 && row < 3) {  // Only subtract in the last column, first 3 rows
                subtract_val = transformed_joint[row];
            }
            // Note: row 3, col 3 (bottom-right) should NOT be subtracted from
            
            rel_transforms[transform_base + i] = transform_chain[j * 16 + i] - subtract_val;
        }
    }
}

// LBS kernel
__global__ void lbs_kernel(
    const float* __restrict__ vertices,    // [B, V, 3]
    const float* __restrict__ weights,     // [V, J]
    const float* __restrict__ transforms,  // [B, J, 4, 4]
    float* __restrict__ posed_vertices,    // [B, V, 3]
    int B, int V, int J
) {
    int batch_idx = blockIdx.x;
    int vertex_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= B || vertex_idx >= V) return;
    
    // Load vertex coordinates
    float vertex[3];
    int v_base = batch_idx * V * 3 + vertex_idx * 3;
    vertex[0] = vertices[v_base];
    vertex[1] = vertices[v_base + 1];
    vertex[2] = vertices[v_base + 2];
    
    // Blend transformations
    float blended_vertex[3] = {0.0f, 0.0f, 0.0f};
    
    for (int j = 0; j < J; j++) {
        float weight = weights[vertex_idx * J + j];
        if (weight > 1e-8f) {
            // Transform vertex by joint j
            int transform_base = batch_idx * J * 16 + j * 16;
            
            // Apply 4x4 transformation: T * [x, y, z, 1]
            float transformed[3];
            for (int i = 0; i < 3; i++) {
                transformed[i] = transforms[transform_base + i * 4 + 3];  // Translation part
                for (int k = 0; k < 3; k++) {
                    transformed[i] += transforms[transform_base + i * 4 + k] * vertex[k];
                }
            }
            
            // Accumulate weighted transformation
            blended_vertex[0] += weight * transformed[0];
            blended_vertex[1] += weight * transformed[1];
            blended_vertex[2] += weight * transformed[2];
        }
    }
    
    // Store result
    posed_vertices[v_base] = blended_vertex[0];
    posed_vertices[v_base + 1] = blended_vertex[1];
    posed_vertices[v_base + 2] = blended_vertex[2];
}

// Host function that returns both posed joints and relative transforms
std::vector<torch::Tensor> batch_rigid_transform_cuda(
    torch::Tensor rot_mats,     // [B, J, 3, 3]
    torch::Tensor joints,       // [B, J, 3]  
    torch::Tensor parents       // [J]
) {
    const int B = rot_mats.size(0);
    const int J = rot_mats.size(1);
    
    // Create output tensors
    auto posed_joints = torch::zeros_like(joints);
    auto rel_transforms = torch::zeros({B, J, 4, 4}, rot_mats.options());
    
    // Calculate shared memory size: 2 * J * 16 floats (transforms_mat + transform_chain)
    size_t shared_mem_size = 2 * J * 16 * sizeof(float);
    
    // Launch kernel - one block per batch, up to 256 threads per block
    dim3 grid(B);
    dim3 block(min(256, max(32, J)));  // At least 32 threads, at most 256
    
    batch_rigid_transform_kernel<<<grid, block, shared_mem_size>>>(
        rot_mats.data_ptr<float>(),
        joints.data_ptr<float>(),
        parents.data_ptr<float>(),
        posed_joints.data_ptr<float>(),
        rel_transforms.data_ptr<float>(),
        B, J
    );
    
    cudaDeviceSynchronize();
    return {posed_joints, rel_transforms};
}

torch::Tensor lbs_cuda(
    torch::Tensor vertices,     // [B, V, 3]
    torch::Tensor weights,      // [V, J]
    torch::Tensor transforms    // [B, J, 4, 4]
) {
    const int B = vertices.size(0);
    const int V = vertices.size(1);
    const int J = weights.size(1);
    
    // Create output tensor [B, V, 3]
    auto posed_vertices = torch::zeros_like(vertices);
    
    // Launch kernel
    dim3 grid(B, (V + 255) / 256);
    dim3 block(1, 256);  // 256 vertices per block
    
    lbs_kernel<<<grid, block>>>(
        vertices.data_ptr<float>(),
        weights.data_ptr<float>(),
        transforms.data_ptr<float>(),
        posed_vertices.data_ptr<float>(),
        B, V, J
    );
    
    cudaDeviceSynchronize();
    return posed_vertices;
}
