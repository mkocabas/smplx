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
    float* __restrict__ transform_chain_out, // [B, J, 4, 4] output
    int B, int J
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;
    
    extern __shared__ float shared_mem[];
    
    float* transforms_mat = shared_mem;
    float* transform_chain = shared_mem + J * 16;
    
    int tid = threadIdx.x;
    
    // Step 1: Compute relative joints and local transformation matrices
    for (int j = tid; j < J; j += blockDim.x) {
        int joint_base = batch_idx * J * 3 + j * 3;
        int rot_base = batch_idx * J * 9 + j * 9;
        int transform_base = j * 16;
        
        float joint_pos[3] = {joints[joint_base], joints[joint_base + 1], joints[joint_base + 2]};
        
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
        
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                transforms_mat[transform_base + i * 4 + k] = rot_mats[rot_base + i * 3 + k];
            }
            transforms_mat[transform_base + i * 4 + 3] = rel_joint[i];
        }
        transforms_mat[transform_base + 12] = 0.0f;
        transforms_mat[transform_base + 13] = 0.0f;
        transforms_mat[transform_base + 14] = 0.0f;
        transforms_mat[transform_base + 15] = 1.0f;
    }
    
    __syncthreads();
    
    // Step 2: Compute kinematic chain
    if (tid == 0) {
        for (int i = 0; i < 16; i++) transform_chain[i] = transforms_mat[i];
        for (int j = 1; j < J; j++) {
            int parent_idx = (int)parents[j];
            matmul_4x4(&transform_chain[parent_idx * 16], &transforms_mat[j * 16], &transform_chain[j * 16]);
        }
    }
    
    __syncthreads();
    
    // Step 3: Extract outputs
    for (int j = tid; j < J; j += blockDim.x) {
        int joint_base = batch_idx * J * 3 + j * 3;
        int transform_base = batch_idx * J * 16 + j * 16;
        
        for(int i=0; i<16; ++i) transform_chain_out[transform_base + i] = transform_chain[j * 16 + i];

        posed_joints[joint_base]     = transform_chain[j * 16 + 3];
        posed_joints[joint_base + 1] = transform_chain[j * 16 + 7];
        posed_joints[joint_base + 2] = transform_chain[j * 16 + 11];
        
        float joint_homogen[4] = {joints[joint_base], joints[joint_base + 1], joints[joint_base + 2], 0.0f};
        
        float transformed_joint[4];
        for (int i = 0; i < 4; i++) {
            transformed_joint[i] = 0.0f;
            for (int k = 0; k < 4; k++) {
                transformed_joint[i] += transform_chain[j * 16 + i * 4 + k] * joint_homogen[k];
            }
        }
        
        for (int i = 0; i < 16; i++) {
            int row = i / 4;
            int col = i % 4;
            float subtract_val = (col == 3 && row < 3) ? transformed_joint[row] : 0.0f;
            rel_transforms[transform_base + i] = transform_chain[j * 16 + i] - subtract_val;
        }
    }
}

// LBS kernel
__global__ void lbs_kernel(
    const float* __restrict__ vertices, const float* __restrict__ weights, const float* __restrict__ transforms,
    float* __restrict__ posed_vertices, int B, int V, int J
) {
    int batch_idx = blockIdx.x;
    int vertex_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= B || vertex_idx >= V) return;
    
    int v_base = batch_idx * V * 3 + vertex_idx * 3;
    float vertex[3] = {vertices[v_base], vertices[v_base + 1], vertices[v_base + 2]};
    
    float blended_vertex[3] = {0.0f, 0.0f, 0.0f};
    
    for (int j = 0; j < J; j++) {
        float weight = weights[vertex_idx * J + j];
        if (weight > 1e-8f) {
            int transform_base = batch_idx * J * 16 + j * 16;
            float transformed[3];
            for (int i = 0; i < 3; i++) {
                transformed[i] = transforms[transform_base + i * 4 + 3];
                for (int k = 0; k < 3; k++) {
                    transformed[i] += transforms[transform_base + i * 4 + k] * vertex[k];
                }
            }
            blended_vertex[0] += weight * transformed[0];
            blended_vertex[1] += weight * transformed[1];
            blended_vertex[2] += weight * transformed[2];
        }
    }
    
    posed_vertices[v_base] = blended_vertex[0];
    posed_vertices[v_base + 1] = blended_vertex[1];
    posed_vertices[v_base + 2] = blended_vertex[2];
}

// Host functions
std::vector<torch::Tensor> batch_rigid_transform_cuda(torch::Tensor r, torch::Tensor j, torch::Tensor p) {
    const int B = r.size(0), J = r.size(1);
    auto posed_j = torch::zeros_like(j);
    auto rel_t = torch::zeros({B, J, 4, 4}, r.options());
    auto chain = torch::zeros({B, J, 4, 4}, r.options());
    batch_rigid_transform_kernel<<<B, min(256, max(32, J)), 2 * J * 16 * sizeof(float)>>>(
        r.data_ptr<float>(), j.data_ptr<float>(), p.data_ptr<float>(),
        posed_j.data_ptr<float>(), rel_t.data_ptr<float>(), chain.data_ptr<float>(), B, J);
    return {posed_j, rel_t, chain};
}

torch::Tensor lbs_cuda(torch::Tensor v, torch::Tensor w, torch::Tensor t) {
    const int B = v.size(0), V = v.size(1);
    auto posed_v = torch::zeros_like(v);
    lbs_kernel<<<dim3(B, (V + 255) / 256), dim3(1, 256)>>>(
        v.data_ptr<float>(), w.data_ptr<float>(), t.data_ptr<float>(), posed_v.data_ptr<float>(),
        B, V, w.size(1));
    return posed_v;
}

// Backward pass kernels
__global__ void lbs_backward_kernel(
    const float* g_v_p, const float* v, const float* w, const float* t,
    float* g_v, float* g_w, float* g_t, int B, int V, int J
) {
    int batch_idx = blockIdx.x, vertex_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= B || vertex_idx >= V) return;

    int v_base = batch_idx * V * 3 + vertex_idx * 3;
    float grad_p_v[3] = {g_v_p[v_base], g_v_p[v_base+1], g_v_p[v_base+2]};
    float vertex[3] = {v[v_base], v[v_base+1], v[v_base+2]};
    
    for (int j = 0; j < J; j++) {
        float weight = w[vertex_idx * J + j];
        if (weight > 1e-8f) {
            int t_base = batch_idx * J * 16 + j * 16;
            for (int i=0; i<3; i++) for (int k=0; k<3; k++) atomicAdd(&g_v[v_base+k], grad_p_v[i] * weight * t[t_base + i*4+k]);
            
            float transformed[3];
            for (int i=0; i<3; i++) {
                transformed[i] = t[t_base + i*4+3];
                for (int k=0; k<3; k++) transformed[i] += t[t_base + i*4+k] * vertex[k];
            }
            float grad_w = 0;
            for (int i=0; i<3; i++) grad_w += grad_p_v[i] * transformed[i];
            atomicAdd(&g_w[vertex_idx*J+j], grad_w);
            
            for (int i=0; i<3; i++) {
                for (int k=0; k<3; k++) atomicAdd(&g_t[t_base+i*4+k], grad_p_v[i] * weight * vertex[k]);
                atomicAdd(&g_t[t_base+i*4+3], grad_p_v[i] * weight);
            }
        }
    }
}

__global__ void batch_rigid_transform_backward_kernel(
    const float* g_p_j, const float* g_r_t, const float* r_m, const float* j_in, const float* p_in,
    const float* t_c, float* g_r_m, float* g_j, int B, int J
) {
    int b_idx = blockIdx.x;
    if (b_idx >= B) return;

    extern __shared__ float shared_mem[];
    float* grad_chain = shared_mem;
    float* local_trans = shared_mem + J * 16;

    for (int i = threadIdx.x; i < J * 16; i += blockDim.x) grad_chain[i] = 0.0f;
    
    for (int j = threadIdx.x; j < J; j += blockDim.x) {
        int j_base = b_idx*J*3+j*3, t_base = b_idx*J*16+j*16, g_t_base = j*16;
        for (int i=0; i<3; i++) grad_chain[g_t_base+i*4+3] += g_p_j[j_base+i];
        for (int i=0; i<16; i++) grad_chain[g_t_base+i] += g_r_t[t_base+i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = J - 1; j > 0; --j) {
            int p_idx = (int)p_in[j];
            float grad_child[16], local_T[16], temp[16];
            for(int i=0; i<16; i++) grad_child[i] = grad_chain[j*16+i];
            
            int j_base = b_idx*J*3+j*3, r_base = b_idx*J*9+j*9;
            float rel_j[3];
            int p_base = b_idx*J*3+p_idx*3;
            rel_j[0] = j_in[j_base] - j_in[p_base];
            rel_j[1] = j_in[j_base+1] - j_in[p_base+1];
            rel_j[2] = j_in[j_base+2] - j_in[p_base+2];

            for(int i=0; i<3; i++) {
                for(int k=0; k<3; k++) local_T[i*4+k] = r_m[r_base+i*3+k];
                local_T[i*4+3] = rel_j[i];
            }
            local_T[12]=local_T[13]=local_T[14]=0; local_T[15]=1;

            float local_T_T[16];
            for(int r=0; r<4; r++) for(int c=0; c<4; c++) local_T_T[r*4+c] = local_T[c*4+r];
            
            matmul_4x4(grad_child, local_T_T, temp);
            for(int i=0; i<16; i++) atomicAdd(&grad_chain[p_idx*16+i], temp[i]);
        }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < J; j += blockDim.x) {
        int p_idx = (j > 0) ? (int)p_in[j] : -1;
        float p_trans_T[16];
        if (p_idx != -1) {
            const float* p_trans = &t_c[b_idx*J*16 + p_idx*16];
            for(int r=0; r<4; r++) for(int c=0; c<4; c++) p_trans_T[r*4+c] = p_trans[c*4+r];
        } else {
            for(int i=0; i<16; i++) p_trans_T[i] = (i%5==0);
        }
        
        float grad_local[16], temp[16];
        matmul_4x4(p_trans_T, &grad_chain[j*16], temp);
        
        int r_base = b_idx*J*9+j*9, j_base = b_idx*J*3+j*3;
        for(int i=0; i<3; i++) {
            for(int k=0; k<3; k++) g_r_m[r_base+i*3+k] = temp[i*4+k];
            atomicAdd(&g_j[j_base+i], temp[i*4+3]);
            if (p_idx != -1) {
                int p_base = b_idx*J*3+p_idx*3;
                atomicAdd(&g_j[p_base+i], -temp[i*4+3]);
            }
        }
    }
}

// Host functions for backward pass
std::vector<torch::Tensor> lbs_backward_cuda(torch::Tensor g_v_p, torch::Tensor v, torch::Tensor w, torch::Tensor t) {
    const int B = v.size(0), V = v.size(1), J = w.size(1);
    auto g_v = torch::zeros_like(v);
    auto g_w = torch::zeros_like(w);
    auto g_t = torch::zeros_like(t);
    lbs_backward_kernel<<<dim3(B, (V+255)/256), dim3(1, 256)>>>(
        g_v_p.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), t.data_ptr<float>(),
        g_v.data_ptr<float>(), g_w.data_ptr<float>(), g_t.data_ptr<float>(), B, V, J);
    return {g_v, g_w, g_t};
}

std::vector<torch::Tensor> batch_rigid_transform_backward_cuda(
    torch::Tensor g_p_j, torch::Tensor g_r_t, torch::Tensor r_m, torch::Tensor j_in,
    torch::Tensor p_in, torch::Tensor t_c
) {
    const int B = r_m.size(0), J = r_m.size(1);
    auto g_r_m = torch::zeros_like(r_m);
    auto g_j = torch::zeros_like(j_in);
    batch_rigid_transform_backward_kernel<<<B, min(256, max(32, J)), 2*J*16*sizeof(float)>>>(
        g_p_j.data_ptr<float>(), g_r_t.data_ptr<float>(), r_m.data_ptr<float>(), j_in.data_ptr<float>(),
        p_in.data_ptr<float>(), t_c.data_ptr<float>(), g_r_m.data_ptr<float>(), g_j.data_ptr<float>(), B, J);
    return {g_r_m, g_j};
}

// Combined LBS forward pass
torch::Tensor lbs_forward_cuda(
    torch::Tensor vertices, torch::Tensor weights, torch::Tensor rot_mats,
    torch::Tensor joints, torch::Tensor parents
) {
    auto results = batch_rigid_transform_cuda(rot_mats, joints, parents);
    return lbs_cuda(vertices, weights, results[1]);
}
