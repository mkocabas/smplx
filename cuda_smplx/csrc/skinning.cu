#include "skinning.h"
#include <torch/types.h> // For CUDAGuard
#include <vector>

// Kernel for calculate_pose_offsets
// pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
// pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, num_vertices, 3)
__global__ void calculate_pose_offsets_kernel(
    const float* __restrict__ rot_mats_data,      // (batch_size, num_joints, 3, 3)
    const float* __restrict__ posedirs_data,      // (P, V3) where P=(num_joints-1)*9, V3=num_vertices*3
    float* pose_offsets_data,                     // (batch_size, num_vertices, 3)
    int batch_size,
    int num_joints,         // Total number of joints (J)
    int num_pose_joints,    // Number of joints used for pose blendshapes (J-1)
    int P_dim,              // Dimension of pose_feature vector, (num_pose_joints * 9)
    int V_dim,              // num_vertices
    int V3_dim             // num_vertices * 3
) {
    // Grid: (batch_size, num_vertices)
    // Threads: (1, 1) - each thread handles one vertex, loops XYZ internally
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y; // Vertex index

    if (b >= batch_size || v_idx >= V_dim) {
        return;
    }

    float ident[9] = {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f};
    
    for (int d_idx = 0; d_idx < 3; ++d_idx) { // Loop over dimensions XYZ
        float val = 0.0f;
        for (int p = 0; p < P_dim; ++p) {
            int joint_idx_pose = p / 9; 
            int mat_elem_idx = p % 9;   
            float rot_mat_val = rot_mats_data[b * num_joints * 9 + (joint_idx_pose + 1) * 9 + mat_elem_idx];
            float feature_comp = rot_mat_val - ident[mat_elem_idx];
            val += feature_comp * posedirs_data[p * V3_dim + (v_idx * 3 + d_idx)];
        }
        pose_offsets_data[b * V_dim * 3 + v_idx * 3 + d_idx] = val;
    }
}


// Kernel for the main skinning operation
// v_homo = T @ v_posed_homo
__global__ void skinning_transform_kernel(
    const float* __restrict__ v_posed_data,       // (batch_size, num_vertices, 3)
    const float* __restrict__ lbs_weights_data,   // (num_vertices, num_joints)
    const float* __restrict__ A_global_data,      // (batch_size, num_joints, 4, 4)
    float* verts_data,                            // (batch_size, num_vertices, 3)
    int batch_size,
    int num_vertices,
    int num_joints
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || v_idx >= num_vertices) {
        return;
    }

    float T_v[16] = {0.0f}; 
    for (int j_idx = 0; j_idx < num_joints; ++j_idx) {
        float weight = lbs_weights_data[v_idx * num_joints + j_idx];
        if (weight == 0.0f) continue; 
        const float* A_global_bj_ptr = A_global_data + b * num_joints * 16 + j_idx * 16;
        for (int i = 0; i < 16; ++i) {
            T_v[i] += weight * A_global_bj_ptr[i];
        }
    }

    float v_posed_homo[4];
    const float* v_posed_bv_ptr = v_posed_data + b * num_vertices * 3 + v_idx * 3;
    v_posed_homo[0] = v_posed_bv_ptr[0];
    v_posed_homo[1] = v_posed_bv_ptr[1];
    v_posed_homo[2] = v_posed_bv_ptr[2];
    v_posed_homo[3] = 1.0f;

    float v_homo[4];
    for (int r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) {
            sum += T_v[r * 4 + c] * v_posed_homo[c];
        }
        v_homo[r] = sum;
    }

    float* verts_bv_ptr = verts_data + b * num_vertices * 3 + v_idx * 3;
    verts_bv_ptr[0] = v_homo[0];
    verts_bv_ptr[1] = v_homo[1];
    verts_bv_ptr[2] = v_homo[2];
}


torch::Tensor calculate_pose_offsets_cuda(
    torch::Tensor rot_mats,
    torch::Tensor posedirs
) {
    TORCH_CHECK(rot_mats.is_cuda() && posedirs.is_cuda());
    TORCH_CHECK(rot_mats.is_contiguous() && posedirs.is_contiguous());
    TORCH_CHECK(rot_mats.scalar_type() == torch::kFloat32 && posedirs.scalar_type() == torch::kFloat32);

    const int batch_size = rot_mats.size(0);
    const int num_joints = rot_mats.size(1); 
    TORCH_CHECK(rot_mats.dim() == 4 && rot_mats.size(0) == batch_size && rot_mats.size(1) == num_joints && rot_mats.size(2)==3 && rot_mats.size(3)==3, "rot_mats shape B,J,3,3");

    const int P_dim = posedirs.size(0); 
    const int V3_dim = posedirs.size(1); 
    TORCH_CHECK(posedirs.dim() == 2, "posedirs shape P, V*3");

    const int num_pose_joints = num_joints - 1;
    TORCH_CHECK(P_dim == num_pose_joints * 9, "P_dim inconsistent with num_joints");
    TORCH_CHECK(V3_dim % 3 == 0, "V3_dim must be multiple of 3");
    const int num_vertices = V3_dim / 3;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rot_mats.device());
    torch::Tensor pose_offsets = torch::empty({batch_size, num_vertices, 3}, options);

    const dim3 threads_po(1, 1); 
    const dim3 blocks_po(batch_size, num_vertices);
    
    torch::cuda::CUDAGuard device_guard(rot_mats.device());

    calculate_pose_offsets_kernel<<<blocks_po, threads_po>>>(
        rot_mats.data_ptr<float>(),
        posedirs.data_ptr<float>(),
        pose_offsets.data_ptr<float>(),
        batch_size, num_joints, num_pose_joints, P_dim, num_vertices, V3_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "calculate_pose_offsets_kernel launch failed: ", cudaGetErrorString(err));
    return pose_offsets;
}


torch::Tensor skinning_cuda(
    torch::Tensor v_shaped,
    torch::Tensor rot_mats,      
    torch::Tensor posedirs,      
    torch::Tensor lbs_weights,
    torch::Tensor A_global
) {
    TORCH_CHECK(v_shaped.is_cuda() && rot_mats.is_cuda() && posedirs.is_cuda() && lbs_weights.is_cuda() && A_global.is_cuda());
    TORCH_CHECK(v_shaped.is_contiguous() && rot_mats.is_contiguous() && posedirs.is_contiguous() && lbs_weights.is_contiguous() && A_global.is_contiguous());
    TORCH_CHECK(v_shaped.scalar_type() == torch::kFloat32, "v_shaped must be float32");
    TORCH_CHECK(rot_mats.scalar_type() == torch::kFloat32, "rot_mats must be float32");
    TORCH_CHECK(posedirs.scalar_type() == torch::kFloat32, "posedirs must be float32");
    TORCH_CHECK(lbs_weights.scalar_type() == torch::kFloat32, "lbs_weights must be float32");
    TORCH_CHECK(A_global.scalar_type() == torch::kFloat32, "A_global must be float32");

    torch::Tensor pose_offsets = calculate_pose_offsets_cuda(rot_mats, posedirs);
    torch::Tensor v_posed = v_shaped + pose_offsets;
    TORCH_CHECK(v_posed.is_contiguous(), "v_posed must be contiguous (result of add should be)");

    const int batch_size = v_posed.size(0);
    const int num_vertices = v_posed.size(1);
    TORCH_CHECK(v_posed.dim() == 3 && v_posed.size(0)==batch_size && v_posed.size(1)==num_vertices && v_posed.size(2) == 3, "v_posed shape B,V,3");

    const int num_joints_lbs = lbs_weights.size(1);
    TORCH_CHECK(lbs_weights.dim() == 2 && lbs_weights.size(0) == num_vertices && lbs_weights.size(1) == num_joints_lbs, "lbs_weights shape V,J");
    
    const int num_joints_A = A_global.size(1);
    TORCH_CHECK(A_global.dim() == 4 && A_global.size(0) == batch_size && A_global.size(1) == num_joints_A && A_global.size(2)==4 && A_global.size(3)==4, "A_global shape B,J,4,4");
    TORCH_CHECK(num_joints_lbs == num_joints_A, "lbs_weights and A_global num_joints mismatch");
    const int num_joints = num_joints_lbs;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(v_posed.device());
    torch::Tensor verts = torch::empty({batch_size, num_vertices, 3}, options);

    const dim3 threads_st(1, 1); 
    const dim3 blocks_st(batch_size, num_vertices);

    torch::cuda::CUDAGuard device_guard(v_posed.device());

    skinning_transform_kernel<<<blocks_st, threads_st>>>(
        v_posed.data_ptr<float>(),
        lbs_weights.data_ptr<float>(),
        A_global.data_ptr<float>(),
        verts.data_ptr<float>(),
        batch_size, num_vertices, num_joints
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "skinning_transform_kernel launch failed: ", cudaGetErrorString(err));

    return verts;
}

// --- Backward Pass Kernels and Functions ---

// --- Backward Pass for calculate_pose_offsets ---
__global__ void calc_po_grad_rot_mats_kernel(
    const float* __restrict__ grad_pose_offsets_data, 
    const float* __restrict__ posedirs_data,          
    float* grad_rot_mats_po_data,                     
    int batch_size, int num_joints, int num_pose_joints, 
    int P_dim, int V_dim, int V3_dim
) {
    int b = blockIdx.x;
    int pj_idx = blockIdx.y; 
    int mat_elem = threadIdx.x; 

    if (b >= batch_size || pj_idx >= num_pose_joints || mat_elem >= 9) return;

    float sum_val = 0.0f;
    for (int v = 0; v < V_dim; ++v) {
        for (int d = 0; d < 3; ++d) {
            sum_val += grad_pose_offsets_data[b*V_dim*3 + v*3 + d] * 
                       posedirs_data[(pj_idx*9 + mat_elem)*V3_dim + v*3 + d];
        }
    }
    atomicAdd(grad_rot_mats_po_data + b*num_joints*9 + (pj_idx+1)*9 + mat_elem, sum_val);
}

__global__ void calc_po_grad_posedirs_kernel(
    const float* __restrict__ grad_pose_offsets_data, 
    const float* __restrict__ rot_mats_fwd_data,      
    float* grad_posedirs_data,                        
    int batch_size, int num_joints, int num_pose_joints,
    int P_dim, int V_dim, int V3_dim
) {
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int v3_idx = blockIdx.y * blockDim.y + threadIdx.y; 

    if (p_idx >= P_dim || v3_idx >= V3_dim) return;

    float ident[9] = {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f};
    int pj_idx = p_idx / 9; 
    int mat_elem = p_idx % 9;

    float sum_val = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        float rot_mat_val = rot_mats_fwd_data[b*num_joints*9 + (pj_idx+1)*9 + mat_elem];
        float feature_comp = rot_mat_val - ident[mat_elem];
        sum_val += grad_pose_offsets_data[b*V_dim*3 + v3_idx] * feature_comp;
    }
    grad_posedirs_data[p_idx * V3_dim + v3_idx] = sum_val;
}

std::vector<torch::Tensor> calculate_pose_offsets_backward_cuda(
    torch::Tensor grad_pose_offsets,
    torch::Tensor rot_mats_fwd,
    torch::Tensor posedirs_fwd
) {
    TORCH_CHECK(grad_pose_offsets.is_cuda() && rot_mats_fwd.is_cuda() && posedirs_fwd.is_cuda());
    TORCH_CHECK(grad_pose_offsets.is_contiguous() && rot_mats_fwd.is_contiguous() && posedirs_fwd.is_contiguous());
    TORCH_CHECK(grad_pose_offsets.scalar_type() == torch::kFloat32 && 
                rot_mats_fwd.scalar_type() == torch::kFloat32 && 
                posedirs_fwd.scalar_type() == torch::kFloat32);

    const int batch_size = grad_pose_offsets.size(0);
    const int V_dim = grad_pose_offsets.size(1);
    const int D_dim = grad_pose_offsets.size(2); 
    TORCH_CHECK(D_dim == 3, "grad_pose_offsets last dim must be 3");
    const int V3_dim = V_dim * 3;

    const int num_joints = rot_mats_fwd.size(1);
    TORCH_CHECK(rot_mats_fwd.dim() == 4 && rot_mats_fwd.size(0) == batch_size && rot_mats_fwd.size(1) == num_joints && rot_mats_fwd.size(2) == 3 && rot_mats_fwd.size(3) == 3);
    const int num_pose_joints = num_joints - 1;
    
    const int P_dim = posedirs_fwd.size(0);
    TORCH_CHECK(posedirs_fwd.dim() == 2 && P_dim == num_pose_joints * 9 && posedirs_fwd.size(1) == V3_dim);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_pose_offsets.device());
    torch::Tensor grad_rot_mats_po = torch::zeros_like(rot_mats_fwd, options); 
    torch::Tensor grad_posedirs = torch::empty_like(posedirs_fwd, options);

    torch::cuda::CUDAGuard device_guard(grad_pose_offsets.device());
    cudaError_t err;

    dim3 threads_grm_po(9); 
    dim3 blocks_grm_po(batch_size, num_pose_joints);
    calc_po_grad_rot_mats_kernel<<<blocks_grm_po, threads_grm_po>>>(
        grad_pose_offsets.data_ptr<float>(), posedirs_fwd.data_ptr<float>(), grad_rot_mats_po.data_ptr<float>(),
        batch_size, num_joints, num_pose_joints, P_dim, V_dim, V3_dim
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "calc_po_grad_rot_mats_kernel launch failed: ", cudaGetErrorString(err));

    dim3 threads_gpd(16, 16); 
    dim3 blocks_gpd((P_dim + threads_gpd.x -1)/threads_gpd.x, (V3_dim + threads_gpd.y -1)/threads_gpd.y);
    calc_po_grad_posedirs_kernel<<<blocks_gpd, threads_gpd>>>(
        grad_pose_offsets.data_ptr<float>(), rot_mats_fwd.data_ptr<float>(), grad_posedirs.data_ptr<float>(),
        batch_size, num_joints, num_pose_joints, P_dim, V_dim, V3_dim
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "calc_po_grad_posedirs_kernel launch failed: ", cudaGetErrorString(err));
    
    return {grad_rot_mats_po, grad_posedirs};
}

// --- Backward Pass for skinning_transform_kernel ---
__global__ void skinning_transform_grad_v_posed_kernel(
    const float* __restrict__ grad_verts_data,      
    const float* __restrict__ lbs_weights_data,   
    const float* __restrict__ A_global_fwd_data,  
    float* grad_v_posed_data,                     
    int B, int V, int J
){
    int b = blockIdx.x; int v = blockIdx.y;
    if (b >= B || v >= V) return;

    float T_v[16] = {0.0f};
    for (int j_idx = 0; j_idx < J; ++j_idx) {
        float weight = lbs_weights_data[v * J + j_idx];
        if (weight == 0.0f) continue;
        const float* A_global_bj_ptr = A_global_fwd_data + b * J * 16 + j_idx * 16;
        for (int i = 0; i < 16; ++i) T_v[i] += weight * A_global_bj_ptr[i];
    }

    float dL_dv_xyz[3];
    const float* current_grad_verts = grad_verts_data + b*V*3 + v*3;

    for(int r=0; r<3; ++r) { 
        float sum = 0.0f;
        for(int c=0; c<3; ++c) { 
            sum += T_v[c*4+r] * current_grad_verts[c]; 
        }
        dL_dv_xyz[r] = sum;
    }
    float* current_grad_v_posed = grad_v_posed_data + b*V*3 + v*3;
    current_grad_v_posed[0] = dL_dv_xyz[0];
    current_grad_v_posed[1] = dL_dv_xyz[1];
    current_grad_v_posed[2] = dL_dv_xyz[2];
}

__global__ void skinning_transform_grad_lbs_weights_kernel(
    const float* __restrict__ grad_verts_data,      
    const float* __restrict__ v_posed_fwd_data,     
    const float* __restrict__ A_global_fwd_data,  
    float* grad_lbs_weights_data,                 
    int B, int V, int J
) {
    int v = blockIdx.x; int j_idx = blockIdx.y;
    if (v >= V || j_idx >= J) return;

    float sum_val = 0.0f;
    for (int b = 0; b < B; ++b) {
        const float* A_bj = A_global_fwd_data + b*J*16 + j_idx*16; 
        const float* vp_bv = v_posed_fwd_data + b*V*3 + v*3;       

        float vp_homo[4] = {vp_bv[0], vp_bv[1], vp_bv[2], 1.0f};
        float transformed_vp_homo[4]; 

        for(int r=0; r<4; ++r) {
            float s = 0.0f;
            for(int c=0; c<4; ++c) s += A_bj[r*4+c] * vp_homo[c];
            transformed_vp_homo[r] = s;
        }
        
        const float* gv_bv = grad_verts_data + b*V*3 + v*3; 
        for(int k=0; k<3; ++k) sum_val += gv_bv[k] * transformed_vp_homo[k];
    }
    atomicAdd(grad_lbs_weights_data + v*J + j_idx, sum_val);
}

__global__ void skinning_transform_grad_A_global_kernel(
    const float* __restrict__ grad_verts_data,      
    const float* __restrict__ v_posed_fwd_data,     
    const float* __restrict__ lbs_weights_data,   
    float* grad_A_global_data,                    
    int B, int V, int J
) {
    int b = blockIdx.x; int j_idx = blockIdx.y; int elem_idx = threadIdx.x; 
    if (b >= B || j_idx >= J || elem_idx >= 16) return;

    int r = elem_idx / 4; 
    int c = elem_idx % 4; 

    float sum_val = 0.0f;
    for (int v = 0; v < V; ++v) {
        float weight = lbs_weights_data[v*J + j_idx];
        if (weight == 0.0f) continue;

        const float* gv_bv = grad_verts_data + b*V*3 + v*3;    
        const float* vp_bv = v_posed_fwd_data + b*V*3 + v*3; 
        float vp_homo_c = (c < 3) ? vp_bv[c] : 1.0f;          

        if (r < 3) {
             sum_val += gv_bv[r] * weight * vp_homo_c;
        }
    }
    atomicAdd(grad_A_global_data + b*J*16 + j_idx*16 + elem_idx, sum_val);
}

std::vector<torch::Tensor> skinning_transform_backward_cuda(
    torch::Tensor grad_verts,
    torch::Tensor v_posed_fwd,
    torch::Tensor lbs_weights_fwd,
    torch::Tensor A_global_fwd
) {
    TORCH_CHECK(grad_verts.is_cuda() && v_posed_fwd.is_cuda() && lbs_weights_fwd.is_cuda() && A_global_fwd.is_cuda());
    TORCH_CHECK(grad_verts.is_contiguous() && v_posed_fwd.is_contiguous() && lbs_weights_fwd.is_contiguous() && A_global_fwd.is_contiguous());
    TORCH_CHECK(grad_verts.scalar_type() == torch::kFloat32 && v_posed_fwd.scalar_type() == torch::kFloat32 && 
                lbs_weights_fwd.scalar_type() == torch::kFloat32 && A_global_fwd.scalar_type() == torch::kFloat32);

    const int B = grad_verts.size(0);
    const int V = grad_verts.size(1);
    const int D_gv = grad_verts.size(2);
    const int J = lbs_weights_fwd.size(1);

    TORCH_CHECK(D_gv == 3, "grad_verts last dim must be 3");
    TORCH_CHECK(v_posed_fwd.dim() == 3 && v_posed_fwd.size(0) == B && v_posed_fwd.size(1) == V && v_posed_fwd.size(2) == 3);
    TORCH_CHECK(lbs_weights_fwd.dim() == 2 && lbs_weights_fwd.size(0) == V && lbs_weights_fwd.size(1) == J);
    TORCH_CHECK(A_global_fwd.dim() == 4 && A_global_fwd.size(0) == B && A_global_fwd.size(1) == J && A_global_fwd.size(2) == 4 && A_global_fwd.size(3) == 4);


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_verts.device());
    torch::Tensor grad_v_posed = torch::empty_like(v_posed_fwd, options);
    torch::Tensor grad_lbs_weights = torch::zeros_like(lbs_weights_fwd, options); 
    torch::Tensor grad_A_global = torch::zeros_like(A_global_fwd, options);    

    torch::cuda::CUDAGuard device_guard(grad_verts.device());
    cudaError_t err;

    dim3 blocks_gvp(B, V); dim3 threads_gvp(1); 
    skinning_transform_grad_v_posed_kernel<<<blocks_gvp, threads_gvp>>>(
        grad_verts.data_ptr<float>(), lbs_weights_fwd.data_ptr<float>(), A_global_fwd.data_ptr<float>(),
        grad_v_posed.data_ptr<float>(), B, V, J);
    err = cudaGetLastError(); TORCH_CHECK(err == cudaSuccess, "skinning_transform_grad_v_posed_kernel failed");

    dim3 blocks_glw(V, J); dim3 threads_glw(1);
    skinning_transform_grad_lbs_weights_kernel<<<blocks_glw, threads_glw>>>(
        grad_verts.data_ptr<float>(), v_posed_fwd.data_ptr<float>(), A_global_fwd.data_ptr<float>(),
        grad_lbs_weights.data_ptr<float>(), B, V, J);
    err = cudaGetLastError(); TORCH_CHECK(err == cudaSuccess, "skinning_transform_grad_lbs_weights_kernel failed");
    
    dim3 blocks_gag(B, J); dim3 threads_gag(16); 
    skinning_transform_grad_A_global_kernel<<<blocks_gag, threads_gag>>>(
        grad_verts.data_ptr<float>(), v_posed_fwd.data_ptr<float>(), lbs_weights_fwd.data_ptr<float>(),
        grad_A_global.data_ptr<float>(), B, V, J);
    err = cudaGetLastError(); TORCH_CHECK(err == cudaSuccess, "skinning_transform_grad_A_global_kernel failed");

    return {grad_v_posed, grad_lbs_weights, grad_A_global};
}

std::vector<torch::Tensor> skinning_backward_cuda(
    torch::Tensor grad_verts,
    torch::Tensor v_shaped_fwd,
    torch::Tensor rot_mats_fwd,
    torch::Tensor posedirs_fwd,
    torch::Tensor lbs_weights_fwd,
    torch::Tensor A_global_fwd,
    torch::Tensor v_posed_fwd 
) {
    TORCH_CHECK(grad_verts.is_cuda() && v_shaped_fwd.is_cuda() && rot_mats_fwd.is_cuda() && 
                posedirs_fwd.is_cuda() && lbs_weights_fwd.is_cuda() && A_global_fwd.is_cuda() && v_posed_fwd.is_cuda());
    TORCH_CHECK(grad_verts.is_contiguous() && v_shaped_fwd.is_contiguous() && rot_mats_fwd.is_contiguous() && 
                posedirs_fwd.is_contiguous() && lbs_weights_fwd.is_contiguous() && A_global_fwd.is_contiguous() && v_posed_fwd.is_contiguous());
    TORCH_CHECK(grad_verts.scalar_type() == torch::kFloat32 && v_shaped_fwd.scalar_type() == torch::kFloat32 &&
                rot_mats_fwd.scalar_type() == torch::kFloat32 && posedirs_fwd.scalar_type() == torch::kFloat32 &&
                lbs_weights_fwd.scalar_type() == torch::kFloat32 && A_global_fwd.scalar_type() == torch::kFloat32 &&
                v_posed_fwd.scalar_type() == torch::kFloat32);
    
    const int B = grad_verts.size(0);
    const int V = grad_verts.size(1);
    const int D = grad_verts.size(2);
    const int J = rot_mats_fwd.size(1);
    const int P_dim_posedirs = posedirs_fwd.size(0);

    TORCH_CHECK(D == 3, "Input tensor dimensions (last) must be 3.");
    TORCH_CHECK(v_shaped_fwd.sizes() == torch::IntArrayRef({B,V,D}), "v_shaped_fwd size mismatch");
    TORCH_CHECK(rot_mats_fwd.sizes() == torch::IntArrayRef({B,J,3,3}), "rot_mats_fwd size mismatch");
    TORCH_CHECK(posedirs_fwd.sizes() == torch::IntArrayRef({P_dim_posedirs,V*D}), "posedirs_fwd size mismatch");
    TORCH_CHECK(lbs_weights_fwd.sizes() == torch::IntArrayRef({V,J}), "lbs_weights_fwd size mismatch");
    TORCH_CHECK(A_global_fwd.sizes() == torch::IntArrayRef({B,J,4,4}), "A_global_fwd size mismatch");
    TORCH_CHECK(v_posed_fwd.sizes() == torch::IntArrayRef({B,V,D}), "v_posed_fwd size mismatch");


    std::vector<torch::Tensor> grads_st = skinning_transform_backward_cuda(
        grad_verts, v_posed_fwd, lbs_weights_fwd, A_global_fwd
    );
    torch::Tensor grad_v_posed = grads_st[0]; 
    torch::Tensor grad_lbs_weights = grads_st[1];
    torch::Tensor grad_A_global = grads_st[2];

    torch::Tensor grad_v_shaped = grad_v_posed.clone(); 
    torch::Tensor grad_pose_offsets = grad_v_posed; 

    std::vector<torch::Tensor> grads_po = calculate_pose_offsets_backward_cuda(
        grad_pose_offsets, rot_mats_fwd, posedirs_fwd
    );
    torch::Tensor grad_rot_mats_po = grads_po[0]; 
    torch::Tensor grad_posedirs = grads_po[1];

    return {grad_v_shaped, grad_rot_mats_po, grad_posedirs, grad_lbs_weights, grad_A_global};
}
