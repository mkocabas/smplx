#include "batch_rigid_transform.h"
#include <torch/types.h> // For CUDAGuard
#include <vector>

// Helper function to construct a 4x4 transformation matrix from R and t
__device__ __forceinline__ void rt_to_transform_matrix(
    const float* __restrict__ R, // 3x3 matrix (9 elements)
    const float* __restrict__ t, // 3x1 vector (3 elements)
    float* T                     // 4x4 matrix (16 elements)
) {
    // R is row-major: R00, R01, R02, R10, R11, R12, R20, R21, R22
    // t is t0, t1, t2
    // T is row-major
    T[0] = R[0]; T[1] = R[1]; T[2] = R[2]; T[3] = t[0];
    T[4] = R[3]; T[5] = R[4]; T[6] = R[5]; T[7] = t[1];
    T[8] = R[6]; T[9] = R[7]; T[10] = R[8]; T[11] = t[2];
    T[12] = 0.0f; T[13] = 0.0f; T[14] = 0.0f; T[15] = 1.0f;
}

// Helper function for 4x4 matrix multiplication: C = A @ B
__device__ __forceinline__ void mat_mult_4x4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* C
) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += A[r * 4 + k] * B[k * 4 + c];
            }
            C[r * 4 + c] = sum;
        }
    }
}

__global__ void batch_rigid_transform_kernel(
    const float* __restrict__ rot_mats_data,    // (batch_size, num_joints, 3, 3)
    const float* __restrict__ joints_data,      // (batch_size, num_joints, 3)
    const long long* __restrict__ parents_data, // (num_joints)
    float* posed_joints_data,                   // (batch_size, num_joints, 3)
    float* A_global_data,                       // (batch_size, num_joints, 4, 4)
    int batch_size,
    int num_joints
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index

    if (b >= batch_size) {
        return;
    }
    
    float rel_joints_b[255 * 3]; 
    if (num_joints > 255) { /* Handle error or use dynamic allocation */ return; }
    float transform_chain_b[255 * 16]; 
     if (num_joints > 255) { /* Handle error */ return; }


    const float* current_joints_ptr = joints_data + b * num_joints * 3;
    for (int j = 0; j < num_joints; ++j) {
        rel_joints_b[j * 3 + 0] = current_joints_ptr[j * 3 + 0];
        rel_joints_b[j * 3 + 1] = current_joints_ptr[j * 3 + 1];
        rel_joints_b[j * 3 + 2] = current_joints_ptr[j * 3 + 2];
        if (j > 0) { 
            long long p_idx = parents_data[j];
            if (p_idx >= 0 && p_idx < num_joints) { 
                 rel_joints_b[j * 3 + 0] -= current_joints_ptr[p_idx * 3 + 0];
                 rel_joints_b[j * 3 + 1] -= current_joints_ptr[p_idx * 3 + 1];
                 rel_joints_b[j * 3 + 2] -= current_joints_ptr[p_idx * 3 + 2];
            }
        }
    }

    for (int j = 0; j < num_joints; ++j) {
        const float* current_rot_mat_ptr = rot_mats_data + b * num_joints * 9 + j * 9;
        const float* current_rel_joint_ptr = rel_joints_b + j * 3;
        
        float local_transform_mat[16]; 
        rt_to_transform_matrix(current_rot_mat_ptr, current_rel_joint_ptr, local_transform_mat);

        float* current_global_transform_ptr = transform_chain_b + j * 16;

        if (parents_data[j] == -1) { 
            for(int k=0; k<16; ++k) current_global_transform_ptr[k] = local_transform_mat[k];
        } else {
            long long p_idx = parents_data[j];
             if (p_idx >= 0 && p_idx < num_joints) { 
                const float* parent_global_transform_ptr = transform_chain_b + p_idx * 16;
                mat_mult_4x4(parent_global_transform_ptr, local_transform_mat, current_global_transform_ptr);
             }
        }
    }
    
    for (int j = 0; j < num_joints; ++j) {
        const float* current_global_transform_ptr = transform_chain_b + j * 16;

        posed_joints_data[b * num_joints * 3 + j * 3 + 0] = current_global_transform_ptr[3];
        posed_joints_data[b * num_joints * 3 + j * 3 + 1] = current_global_transform_ptr[7];
        posed_joints_data[b * num_joints * 3 + j * 3 + 2] = current_global_transform_ptr[11];

        float current_joint_orig_loc[3] = {
            current_joints_ptr[j * 3 + 0], 
            current_joints_ptr[j * 3 + 1], 
            current_joints_ptr[j * 3 + 2]
        };

        float R_g_times_joint_orig[3];
        R_g_times_joint_orig[0] = current_global_transform_ptr[0] * current_joint_orig_loc[0] + current_global_transform_ptr[1] * current_joint_orig_loc[1] + current_global_transform_ptr[2] * current_joint_orig_loc[2];
        R_g_times_joint_orig[1] = current_global_transform_ptr[4] * current_joint_orig_loc[0] + current_global_transform_ptr[5] * current_joint_orig_loc[1] + current_global_transform_ptr[6] * current_joint_orig_loc[2];
        R_g_times_joint_orig[2] = current_global_transform_ptr[8] * current_joint_orig_loc[0] + current_global_transform_ptr[9] * current_joint_orig_loc[1] + current_global_transform_ptr[10] * current_joint_orig_loc[2];

        float* current_A_global_ptr = A_global_data + b * num_joints * 16 + j * 16;
        for(int k=0; k<12; ++k) current_A_global_ptr[k] = current_global_transform_ptr[k]; 

        current_A_global_ptr[3] = current_global_transform_ptr[3] - R_g_times_joint_orig[0];
        current_A_global_ptr[7] = current_global_transform_ptr[7] - R_g_times_joint_orig[1];
        current_A_global_ptr[11] = current_global_transform_ptr[11] - R_g_times_joint_orig[2];
        
        current_A_global_ptr[12] = 0.0f; current_A_global_ptr[13] = 0.0f; current_A_global_ptr[14] = 0.0f; current_A_global_ptr[15] = 1.0f;
    }
}


std::vector<torch::Tensor> batch_rigid_transform_cuda(
    torch::Tensor rot_mats,
    torch::Tensor joints,
    torch::Tensor parents
) {
    TORCH_CHECK(rot_mats.is_cuda(), "rot_mats must be a CUDA tensor");
    TORCH_CHECK(joints.is_cuda(), "joints must be a CUDA tensor");
    // TORCH_CHECK(parents.is_cpu(), "parents must be a CPU tensor for this kernel (or pass device then copy inside)"); 
    // Updated: parents is moved to GPU inside this wrapper.

    TORCH_CHECK(rot_mats.is_contiguous(), "rot_mats must be contiguous");
    TORCH_CHECK(joints.is_contiguous(), "joints must be contiguous");
    TORCH_CHECK(parents.is_contiguous(), "parents must be contiguous");

    TORCH_CHECK(rot_mats.scalar_type() == torch::kFloat32, "rot_mats must be float32");
    TORCH_CHECK(joints.scalar_type() == torch::kFloat32, "joints must be float32");
    TORCH_CHECK(parents.scalar_type() == torch::kInt64, "parents must be int64");

    TORCH_CHECK(rot_mats.dim() == 4 && rot_mats.size(2) == 3 && rot_mats.size(3) == 3, "rot_mats shape (B, J, 3, 3)");
    TORCH_CHECK(joints.dim() == 3 && joints.size(2) == 3, "joints shape (B, J, 3)");
    TORCH_CHECK(parents.dim() == 1, "parents shape (J)");

    const int batch_size = rot_mats.size(0);
    const int num_joints_rot_mats = rot_mats.size(1);
    const int num_joints_joints = joints.size(1); // Corrected from prompt's joints.size(0)
    TORCH_CHECK(num_joints_joints == num_joints_rot_mats, "rot_mats and joints must have same num_joints");
    TORCH_CHECK(parents.size(0) == num_joints_rot_mats, "parents and rot_mats must have same num_joints");
    
    const int num_joints = num_joints_rot_mats;
    TORCH_CHECK(num_joints <= 255, "Number of joints exceeds kernel static array limit of 255.");


    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(rot_mats.device());
    torch::Tensor posed_joints = torch::empty({batch_size, num_joints, 3}, options_float);
    torch::Tensor A_global = torch::empty({batch_size, num_joints, 4, 4}, options_float);
    
    torch::Tensor parents_gpu = parents.to(rot_mats.device(), /*non_blocking=*/false); 
    if (!parents_gpu.is_contiguous()) { // Ensure contiguity after potential copy
        parents_gpu = parents_gpu.contiguous();
    }


    const int threads_per_block = 64; 
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    torch::cuda::CUDAGuard device_guard(rot_mats.device());

    batch_rigid_transform_kernel<<<blocks, threads_per_block>>>(
        rot_mats.data_ptr<float>(),
        joints.data_ptr<float>(),
        parents_gpu.data_ptr<long long>(), 
        posed_joints.data_ptr<float>(),
        A_global.data_ptr<float>(),
        batch_size,
        num_joints
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel (batch_rigid_transform_kernel) launch failed with error ", cudaGetErrorString(err));
    
    return {posed_joints, A_global};
}

// --- Backward Pass Kernels and Function ---

// Helper to form Tj_rest = [I | J_fwd_j]
__device__ __forceinline__ void get_Tj_rest(const float* J_fwd_j_ptr, float* Tj_rest_ptr) {
    // J_fwd_j_ptr is (3)
    // Tj_rest_ptr is (4,4) = 16 elements
    Tj_rest_ptr[0]=1.0f; Tj_rest_ptr[1]=0.0f; Tj_rest_ptr[2]=0.0f; Tj_rest_ptr[3]=J_fwd_j_ptr[0];
    Tj_rest_ptr[4]=0.0f; Tj_rest_ptr[5]=1.0f; Tj_rest_ptr[6]=0.0f; Tj_rest_ptr[7]=J_fwd_j_ptr[1];
    Tj_rest_ptr[8]=0.0f; Tj_rest_ptr[9]=0.0f; Tj_rest_ptr[10]=1.0f; Tj_rest_ptr[11]=J_fwd_j_ptr[2];
    Tj_rest_ptr[12]=0.0f; Tj_rest_ptr[13]=0.0f; Tj_rest_ptr[14]=0.0f; Tj_rest_ptr[15]=1.0f;
}

// Helper: Transpose of a 4x4 matrix
__device__ __forceinline__ void transpose_4x4(const float* A, float* A_T) {
    for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) A_T[c*4+r] = A[r*4+c];
}

// Helper: 3x3 matrix vector multiplication: out_vec = mat * vec (from batch_rodrigues.cu)
__device__ __forceinline__ void mat_vec_mult_3x1(const float* __restrict__ mat, const float* __restrict__ vec, float* out_vec) {
    for(int r=0; r<3; ++r) {
        float sum = 0.0f;
        for(int c=0; c<3; ++c) sum += mat[r*3+c] * vec[c];
        out_vec[r] = sum;
    }
}


__global__ void batch_rigid_transform_backward_kernel(
    const float* __restrict__ grad_posed_joints_data, // (B,J,3)
    const float* __restrict__ grad_A_global_data,     // (B,J,4,4)
    const float* __restrict__ rot_mats_fwd_data,      // (B,J,3,3)
    const float* __restrict__ joints_fwd_data,        // (B,J,3) (J_shaped)
    const long long* __restrict__ parents_fwd_data,   // (J) (GPU copy)
    const float* __restrict__ A_global_fwd_data,      // (B,J,4,4) 

    float* grad_rot_mats_data,                       // (B,J,3,3) (Output)
    float* grad_joints_data,                         // (B,J,3) (Output, for J_shaped)
    int batch_size, int num_joints
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    if (b >= batch_size) return;

    float grad_G_j_accum[255 * 16]; 
    if (num_joints > 255) { return; }
    for(int i=0; i < num_joints * 16; ++i) grad_G_j_accum[i] = 0.0f;
    
    float grad_T_local_j_accum[255*16]; 
    if (num_joints > 255) { return; }
    for(int i=0; i < num_joints * 16; ++i) grad_T_local_j_accum[i] = 0.0f;


    for (int j = 0; j < num_joints; ++j) {
        const float* current_A_global_fwd_ptr = A_global_fwd_data + b*num_joints*16 + j*16;
        const float* current_joints_fwd_ptr = joints_fwd_data + b*num_joints*3 + j*3;
        const float* current_grad_A_global_ptr = grad_A_global_data + b*num_joints*16 + j*16;
        float* current_grad_G_ptr = grad_G_j_accum + j*16;

        float Tj_rest[16], Tj_rest_T[16];
        get_Tj_rest(current_joints_fwd_ptr, Tj_rest);
        transpose_4x4(Tj_rest, Tj_rest_T);
        
        mat_mult_4x4(current_grad_A_global_ptr, Tj_rest_T, current_grad_G_ptr);

        const float* current_grad_posed_joints_ptr = grad_posed_joints_data + b*num_joints*3 + j*3;
        current_grad_G_ptr[3]  += current_grad_posed_joints_ptr[0]; 
        current_grad_G_ptr[7]  += current_grad_posed_joints_ptr[1]; 
        current_grad_G_ptr[11] += current_grad_posed_joints_ptr[2]; 
    }

    float G_fwd_b[255*16], T_local_fwd_b[255*16];
    if (num_joints > 255) return;
    float rel_joints_b[255*3]; 
    if (num_joints > 255) return;

    const float* current_J_fwd_b_ptr = joints_fwd_data + b * num_joints * 3;
     for (int j = 0; j < num_joints; ++j) {
        rel_joints_b[j*3+0] = current_J_fwd_b_ptr[j*3+0];
        rel_joints_b[j*3+1] = current_J_fwd_b_ptr[j*3+1];
        rel_joints_b[j*3+2] = current_J_fwd_b_ptr[j*3+2];
        if (j > 0) {
            long long p_idx = parents_fwd_data[j];
            if (p_idx >=0 && p_idx < num_joints){ // Bounds check
                rel_joints_b[j*3+0] -= current_J_fwd_b_ptr[p_idx*3+0];
                rel_joints_b[j*3+1] -= current_J_fwd_b_ptr[p_idx*3+1];
                rel_joints_b[j*3+2] -= current_J_fwd_b_ptr[p_idx*3+2];
            }
        }
        const float* current_rot_mat_fwd_ptr = rot_mats_fwd_data + b*num_joints*9 + j*9;
        rt_to_transform_matrix(current_rot_mat_fwd_ptr, rel_joints_b + j*3, T_local_fwd_b + j*16);
    }
    for (int j=0; j<num_joints; ++j) {
        if (parents_fwd_data[j] == -1) { 
            for(int k=0; k<16; ++k) G_fwd_b[j*16+k] = T_local_fwd_b[j*16+k];
        } else {
            long long p_idx = parents_fwd_data[j];
             if (p_idx >=0 && p_idx < num_joints){ // Bounds check
                mat_mult_4x4(G_fwd_b + p_idx*16, T_local_fwd_b + j*16, G_fwd_b + j*16);
             }
        }
    }
    
    for (int j = num_joints - 1; j >= 0; --j) {
        const float* current_grad_G_ptr = grad_G_j_accum + j*16;
        float* current_grad_T_local_ptr = grad_T_local_j_accum + j*16; 

        if (parents_fwd_data[j] == -1) { 
             for(int k=0; k<16; ++k) current_grad_T_local_ptr[k] += current_grad_G_ptr[k];
        } else {
            long long p_idx = parents_fwd_data[j];
            if (p_idx < 0 || p_idx >= num_joints) continue; // Invalid parent
            
            const float* G_parent_fwd_ptr = G_fwd_b + p_idx*16;
            const float* T_local_fwd_ptr = T_local_fwd_b + j*16;
            
            float* grad_G_parent_accum_ptr = grad_G_j_accum + p_idx*16; 

            float T_local_fwd_T[16];
            transpose_4x4(T_local_fwd_ptr, T_local_fwd_T);
            float term_for_grad_G_parent[16];
            mat_mult_4x4(current_grad_G_ptr, T_local_fwd_T, term_for_grad_G_parent);
            for(int k=0; k<16; ++k) grad_G_parent_accum_ptr[k] += term_for_grad_G_parent[k];

            float G_parent_fwd_T[16];
            transpose_4x4(G_parent_fwd_ptr, G_parent_fwd_T);
            float term_for_grad_T_local[16];
            mat_mult_4x4(G_parent_fwd_T, current_grad_G_ptr, term_for_grad_T_local);
             for(int k=0; k<16; ++k) current_grad_T_local_ptr[k] += term_for_grad_T_local[k];
        }
    }

    float* current_grad_joints_ptr_b_output = grad_joints_data + b*num_joints*3; 
    for(int j=0; j<num_joints*3; ++j) current_grad_joints_ptr_b_output[j] = 0.0f; 

    for (int j = 0; j < num_joints; ++j) {
        const float* current_grad_T_local_ptr = grad_T_local_j_accum + j*16;
        float* current_grad_rot_mat_ptr = grad_rot_mats_data + b*num_joints*9 + j*9;
        
        current_grad_rot_mat_ptr[0]=current_grad_T_local_ptr[0]; current_grad_rot_mat_ptr[1]=current_grad_T_local_ptr[1]; current_grad_rot_mat_ptr[2]=current_grad_T_local_ptr[2];
        current_grad_rot_mat_ptr[3]=current_grad_T_local_ptr[4]; current_grad_rot_mat_ptr[4]=current_grad_T_local_ptr[5]; current_grad_rot_mat_ptr[5]=current_grad_T_local_ptr[6];
        current_grad_rot_mat_ptr[6]=current_grad_T_local_ptr[8]; current_grad_rot_mat_ptr[7]=current_grad_T_local_ptr[9]; current_grad_rot_mat_ptr[8]=current_grad_T_local_ptr[10];
        
        float grad_rel_joint_j[3];
        grad_rel_joint_j[0] = current_grad_T_local_ptr[3];
        grad_rel_joint_j[1] = current_grad_T_local_ptr[7];
        grad_rel_joint_j[2] = current_grad_T_local_ptr[11];

        current_grad_joints_ptr_b_output[j*3+0] += grad_rel_joint_j[0];
        current_grad_joints_ptr_b_output[j*3+1] += grad_rel_joint_j[1];
        current_grad_joints_ptr_b_output[j*3+2] += grad_rel_joint_j[2];
        if (j > 0) {
            long long p_idx = parents_fwd_data[j];
            if (p_idx >=0 && p_idx < num_joints){ // Bounds check
                current_grad_joints_ptr_b_output[p_idx*3+0] -= grad_rel_joint_j[0];
                current_grad_joints_ptr_b_output[p_idx*3+1] -= grad_rel_joint_j[1];
                current_grad_joints_ptr_b_output[p_idx*3+2] -= grad_rel_joint_j[2];
            }
        }
    }
    
    for (int j = 0; j < num_joints; ++j) {
        const float* current_grad_A_global_ptr = grad_A_global_data + b*num_joints*16 + j*16; 
        const float* G_fwd_bj_ptr = G_fwd_b + j*16; 
        
        float G_rot_T[9]; 
        G_rot_T[0]=G_fwd_bj_ptr[0]; G_rot_T[1]=G_fwd_bj_ptr[4]; G_rot_T[2]=G_fwd_bj_ptr[8];
        G_rot_T[3]=G_fwd_bj_ptr[1]; G_rot_T[4]=G_fwd_bj_ptr[5]; G_rot_T[5]=G_fwd_bj_ptr[9];
        G_rot_T[6]=G_fwd_bj_ptr[2]; G_rot_T[7]=G_fwd_bj_ptr[6]; G_rot_T[8]=G_fwd_bj_ptr[10];

        float dL_dTrans_A[3] = {current_grad_A_global_ptr[3], current_grad_A_global_ptr[7], current_grad_A_global_ptr[11]};
        float grad_neg_Jj[3];
        mat_vec_mult_3x1(G_rot_T, dL_dTrans_A, grad_neg_Jj);

        current_grad_joints_ptr_b_output[j*3+0] -= grad_neg_Jj[0]; 
        current_grad_joints_ptr_b_output[j*3+1] -= grad_neg_Jj[1];
        current_grad_joints_ptr_b_output[j*3+2] -= grad_neg_Jj[2];
    }
}


std::vector<torch::Tensor> batch_rigid_transform_backward_cuda(
    torch::Tensor grad_posed_joints,
    torch::Tensor grad_A_global,
    torch::Tensor rot_mats_fwd,
    torch::Tensor joints_fwd, // J_shaped
    torch::Tensor parents_fwd,
    torch::Tensor posed_joints_fwd, 
    torch::Tensor A_global_fwd
) {
    TORCH_CHECK(grad_posed_joints.is_cuda() && grad_A_global.is_cuda() && rot_mats_fwd.is_cuda() && 
                joints_fwd.is_cuda() && A_global_fwd.is_cuda() && posed_joints_fwd.is_cuda());
    TORCH_CHECK(grad_posed_joints.is_contiguous() && grad_A_global.is_contiguous() && rot_mats_fwd.is_contiguous() && 
                joints_fwd.is_contiguous() && parents_fwd.is_contiguous() && 
                posed_joints_fwd.is_contiguous() && A_global_fwd.is_contiguous());
    
    TORCH_CHECK(grad_posed_joints.scalar_type() == torch::kFloat32);
    TORCH_CHECK(grad_A_global.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rot_mats_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(joints_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(parents_fwd.scalar_type() == torch::kInt64); // parents_fwd is int64
    TORCH_CHECK(posed_joints_fwd.scalar_type() == torch::kFloat32);
    TORCH_CHECK(A_global_fwd.scalar_type() == torch::kFloat32);


    const int batch_size = grad_posed_joints.size(0);
    const int num_joints = grad_posed_joints.size(1);
    
    TORCH_CHECK(grad_posed_joints.dim() == 3 && grad_posed_joints.size(2) == 3);
    TORCH_CHECK(grad_A_global.dim() == 4 && grad_A_global.size(0) == batch_size && grad_A_global.size(1) == num_joints && grad_A_global.size(2) == 4 && grad_A_global.size(3) == 4);
    TORCH_CHECK(rot_mats_fwd.dim() == 4 && rot_mats_fwd.size(0) == batch_size && rot_mats_fwd.size(1) == num_joints && rot_mats_fwd.size(2) == 3 && rot_mats_fwd.size(3) == 3);
    TORCH_CHECK(joints_fwd.dim() == 3 && joints_fwd.size(0) == batch_size && joints_fwd.size(1) == num_joints && joints_fwd.size(2) == 3);
    TORCH_CHECK(parents_fwd.dim() == 1 && parents_fwd.size(0) == num_joints);
    TORCH_CHECK(posed_joints_fwd.dim() == 3 && posed_joints_fwd.size(0) == batch_size && posed_joints_fwd.size(1) == num_joints && posed_joints_fwd.size(2) == 3);
    TORCH_CHECK(A_global_fwd.dim() == 4 && A_global_fwd.size(0) == batch_size && A_global_fwd.size(1) == num_joints && A_global_fwd.size(2) == 4 && A_global_fwd.size(3) == 4);


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(grad_posed_joints.device());
    torch::Tensor grad_rot_mats = torch::zeros_like(rot_mats_fwd, options); 
    torch::Tensor grad_joints = torch::zeros_like(joints_fwd, options);   

    torch::Tensor parents_gpu = parents_fwd.to(rot_mats_fwd.device(), /*non_blocking=*/false).contiguous();
    if (!parents_gpu.is_contiguous()) { parents_gpu = parents_gpu.contiguous(); }


    const int threads_per_block = 64; 
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    torch::cuda::CUDAGuard device_guard(grad_posed_joints.device());

    batch_rigid_transform_backward_kernel<<<blocks, threads_per_block>>>(
        grad_posed_joints.data_ptr<float>(),
        grad_A_global.data_ptr<float>(),
        rot_mats_fwd.data_ptr<float>(),
        joints_fwd.data_ptr<float>(),
        parents_gpu.data_ptr<long long>(),
        A_global_fwd.data_ptr<float>(), // Pass A_global_fwd
        grad_rot_mats.data_ptr<float>(),
        grad_joints.data_ptr<float>(),
        batch_size, num_joints
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_rigid_transform_backward_kernel launch failed: ", cudaGetErrorString(err));

    return {grad_rot_mats, grad_joints, torch::Tensor()};
}
