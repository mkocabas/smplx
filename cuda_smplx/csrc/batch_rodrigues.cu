#include "batch_rodrigues.h"
#include <torch/types.h> // For CUDAGuard
#include <cmath> // For sqrtf, cosf, sinf

// CUDA kernel for batch_rodrigues
__global__ void batch_rodrigues_kernel(
    const float* __restrict__ rot_vecs, // (N, 3)
    float* rot_mats,                    // (N, 3, 3)
    int N,
    float epsilon                       // Small value for stability
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Index for N

    if (i >= N) {
        return;
    }

    const float* rv = rot_vecs + i * 3; // Pointer to current rot_vec rv[0], rv[1], rv[2]
    float* R = rot_mats + i * 9;        // Pointer to current rot_mat R[0]..R[8]

    float angle_sq = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];

    // Check if angle is close to zero
    if (angle_sq < epsilon * epsilon) { // Using epsilon*epsilon for squared comparison
        // Identity matrix for zero rotation
        R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
        R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
        R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
        return;
    }

    float angle = sqrtf(angle_sq);
    float ca = cosf(angle);
    float sa = sinf(angle);

    // Normalized rotation axis
    float rx = rv[0] / angle;
    float ry = rv[1] / angle;
    float rz = rv[2] / angle;

    // Skew-symmetric matrix K
    // K = [ 0, -rz,  ry]
    //     [ rz,  0, -rx]
    //     [-ry,  rx,  0]

    // K^2 = [ -ry^2-rz^2,   rx*ry,        rx*rz     ]
    //       [   rx*ry,    -rx^2-rz^2,     ry*rz     ]
    //       [   rx*rz,      ry*rz,      -rx^2-ry^2  ]
    // which can also be written as (r*r^T - I) using r = [rx,ry,rz]^T (outer product)

    // R = I + sa * K + (1 - ca) * K^2
    // R = I + sa * K + (1 - ca) * (r*r^T - I)
    // R = ca * I + sa * K + (1 - ca) * r*r^T
    
    // Element-wise computation
    R[0] = ca + (1 - ca) * rx * rx;
    R[1] = (1 - ca) * rx * ry - sa * rz;
    R[2] = (1 - ca) * rx * rz + sa * ry;

    R[3] = (1 - ca) * ry * rx + sa * rz;
    R[4] = ca + (1 - ca) * ry * ry;
    R[5] = (1 - ca) * ry * rz - sa * rx;

    R[6] = (1 - ca) * rz * rx - sa * ry;
    R[7] = (1 - ca) * rz * ry + sa * rx;
    R[8] = ca + (1 - ca) * rz * rz;
}


torch::Tensor batch_rodrigues_cuda(
    torch::Tensor rot_vecs
) {
    TORCH_CHECK(rot_vecs.is_cuda(), "rot_vecs must be a CUDA tensor");
    TORCH_CHECK(rot_vecs.is_contiguous(), "rot_vecs must be contiguous");
    TORCH_CHECK(rot_vecs.scalar_type() == torch::kFloat32, "rot_vecs must be a float32 tensor");
    TORCH_CHECK(rot_vecs.dim() == 2, "rot_vecs must be a 2D tensor");
    TORCH_CHECK(rot_vecs.size(1) == 3, "rot_vecs must have 3 columns (axis-angle)");

    const int N = rot_vecs.size(0);
    const float epsilon = 1e-8f; // Epsilon for stability check

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rot_vecs.device());
    torch::Tensor rot_mats = torch::empty({N, 3, 3}, options);

    const int threads_per_block = 256; // Example: 256 threads per block
    const int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    torch::cuda::CUDAGuard device_guard(rot_vecs.device());

    batch_rodrigues_kernel<<<blocks, threads_per_block>>>(
        rot_vecs.data_ptr<float>(),
        rot_mats.data_ptr<float>(),
        N,
        epsilon
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel (batch_rodrigues_kernel) launch failed with error ", cudaGetErrorString(err));

    return rot_mats;
}

// --- Backward Pass Kernels and Function ---

// Helper: Skew-symmetric matrix from vector
__device__ __forceinline__ void vec_to_skew(const float* __restrict__ vec, float* skew_mat) {
    // vec: [v0, v1, v2]
    // skew_mat: [ 0, -v2,  v1]
    //           [ v2,  0, -v0]
    //           [-v1,  v0,  0]
    skew_mat[0] = 0.0f;  skew_mat[1] = -vec[2]; skew_mat[2] = vec[1];
    skew_mat[3] = vec[2]; skew_mat[4] = 0.0f;  skew_mat[5] = -vec[0];
    skew_mat[6] = -vec[1]; skew_mat[7] = vec[0];  skew_mat[8] = 0.0f;
}

// Helper: Vee operator (skew-symmetric matrix to vector)
__device__ __forceinline__ void skew_to_vec(const float* __restrict__ skew_mat, float* vec) {
    // skew_mat: [ 0, -v2,  v1]
    //           [ v2,  0, -v0]
    //           [-v1,  v0,  0]
    // vec: [v0, v1, v2]
    vec[0] = skew_mat[7]; // (2,1) or (-skew_mat[5])
    vec[1] = skew_mat[2]; // (0,2) or (-skew_mat[6])
    vec[2] = skew_mat[3]; // (1,0) or (-skew_mat[1])
}

// Helper: 3x3 matrix multiplication C = A * B
__device__ __forceinline__ void mat_mult_3x3(const float* __restrict__ A, const float* __restrict__ B, float* C) {
    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            float sum = 0.0f;
            for(int k=0; k<3; ++k) sum += A[r*3+k] * B[k*3+c];
            C[r*3+c] = sum;
        }
    }
}

// Helper: 3x3 matrix transpose B = A^T
__device__ __forceinline__ void mat_transpose_3x3(const float* __restrict__ A, float* B) {
    B[0]=A[0]; B[1]=A[3]; B[2]=A[6];
    B[3]=A[1]; B[4]=A[4]; B[5]=A[7];
    B[6]=A[2]; B[7]=A[5]; B[8]=A[8];
}

// Helper: 3x3 matrix subtraction C = A - B
__device__ __forceinline__ void mat_sub_3x3(const float* __restrict__ A, const float* __restrict__ B, float* C) {
    for(int i=0; i<9; ++i) C[i] = A[i] - B[i];
}

// Helper: 3x3 matrix scale C = s * A
__device__ __forceinline__ void mat_scale_3x3(const float* __restrict__ A, float s, float* C) {
    for(int i=0; i<9; ++i) C[i] = s * A[i];
}

// Helper: 3x1 vector u*u^T (outer product) -> 3x3 matrix
__device__ __forceinline__ void vec_outer_prod_3x3(const float* __restrict__ u, float* out_mat) {
    out_mat[0]=u[0]*u[0]; out_mat[1]=u[0]*u[1]; out_mat[2]=u[0]*u[2];
    out_mat[3]=u[1]*u[0]; out_mat[4]=u[1]*u[1]; out_mat[5]=u[1]*u[2];
    out_mat[6]=u[2]*u[0]; out_mat[7]=u[2]*u[1]; out_mat[8]=u[2]*u[2];
}

// Helper: 3x3 matrix addition C = A + B
__device__ __forceinline__ void mat_add_3x3(const float* __restrict__ A, const float* __restrict__ B, float* C) {
    for(int i=0; i<9; ++i) C[i] = A[i] + B[i];
}

// Helper: 3x3 matrix vector multiplication: out_vec = mat * vec
__device__ __forceinline__ void mat_vec_mult_3x1(const float* __restrict__ mat, const float* __restrict__ vec, float* out_vec) {
    for(int r=0; r<3; ++r) {
        float sum = 0.0f;
        for(int c=0; c<3; ++c) sum += mat[r*3+c] * vec[c];
        out_vec[r] = sum;
    }
}


__global__ void batch_rodrigues_backward_kernel(
    const float* __restrict__ grad_rot_mats_data, // (N, 3, 3) dL/dC
    const float* __restrict__ rot_vecs_data,      // (N, 3) phi
    const float* __restrict__ rot_mats_fwd_data,  // (N, 3, 3) C
    float* grad_rot_vecs_data,                    // (N, 3) dL/dphi
    int N,
    float epsilon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* dL_dC = grad_rot_mats_data + i * 9;
    const float* phi   = rot_vecs_data + i * 3;
    const float* C     = rot_mats_fwd_data + i * 9;
    float* dL_dphi = grad_rot_vecs_data + i * 3;

    // 1. A = (dL/dC) * C^T
    float C_T[9];
    mat_transpose_3x3(C, C_T);
    float A[9];
    mat_mult_3x3(dL_dC, C_T, A);

    // 2. A_skew = (A - A^T) / 2
    float A_T[9];
    mat_transpose_3x3(A, A_T);
    float A_minus_AT[9];
    mat_sub_3x3(A, A_T, A_minus_AT);
    float A_skew[9];
    mat_scale_3x3(A_minus_AT, 0.5f, A_skew);

    // 3. dL_skew_vee = vee(A_skew)
    float dL_skew_vee[3];
    skew_to_vec(A_skew, dL_skew_vee);

    // 4. Compute J_L_inv_T (transpose of inverse of Left SO(3) Jacobian)
    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float J_L_inv_T[9]; 

    if (theta_sq < epsilon * epsilon) { 
        // J_L_inv_T approaches I + 0.5 * [phi]_x
        J_L_inv_T[0]=1.0f; J_L_inv_T[1]=0.0f; J_L_inv_T[2]=0.0f;
        J_L_inv_T[3]=0.0f; J_L_inv_T[4]=1.0f; J_L_inv_T[5]=0.0f;
        J_L_inv_T[6]=0.0f; J_L_inv_T[7]=0.0f; J_L_inv_T[8]=1.0f;

        float phi_skew_temp[9]; // Renamed to avoid conflict with phi vector
        vec_to_skew(phi, phi_skew_temp); 
        mat_scale_3x3(phi_skew_temp, 0.5f, phi_skew_temp); 
        mat_add_3x3(J_L_inv_T, phi_skew_temp, J_L_inv_T); 
        
    } else {
        float theta = sqrtf(theta_sq);
        float u[3] = {phi[0]/theta, phi[1]/theta, phi[2]/theta};
        
        float half_theta = theta * 0.5f;
        float cot_half_theta;
        if (fabsf(sinf(half_theta)) < epsilon) { // Avoid division by zero if sin(half_theta) is too small
             cot_half_theta = (half_theta > 0 ? 1.0f : -1.0f) / epsilon ; // Large value
        } else {
             cot_half_theta = cosf(half_theta) / sinf(half_theta);
        }


        float coeff1 = half_theta * cot_half_theta;
        float coeff2 = 1.0f - coeff1;

        float I_mat[9] = {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f};
        float u_ut_mat[9];
        vec_outer_prod_3x3(u, u_ut_mat);
        float u_skew_mat[9];
        vec_to_skew(u, u_skew_mat);

        float term1[9], term2[9], term3[9], temp_sum[9];
        mat_scale_3x3(I_mat, coeff1, term1);
        mat_scale_3x3(u_ut_mat, coeff2, term2);
        mat_scale_3x3(u_skew_mat, -half_theta, term3); // -0.5*theta*[u]_x
        
        mat_add_3x3(term1, term2, temp_sum);
        float J_L_inv[9];
        mat_add_3x3(temp_sum, term3, J_L_inv);
        
        mat_transpose_3x3(J_L_inv, J_L_inv_T);
    }

    // 5. dL/dphi = J_L_inv_T * dL_skew_vee
    mat_vec_mult_3x1(J_L_inv_T, dL_skew_vee, dL_dphi);
}


torch::Tensor batch_rodrigues_backward_cuda(
    torch::Tensor grad_rot_mats,
    torch::Tensor rot_vecs,
    torch::Tensor rot_mats_fwd
) {
    TORCH_CHECK(grad_rot_mats.is_cuda() && rot_vecs.is_cuda() && rot_mats_fwd.is_cuda());
    TORCH_CHECK(grad_rot_mats.is_contiguous() && rot_vecs.is_contiguous() && rot_mats_fwd.is_contiguous());
    TORCH_CHECK(grad_rot_mats.scalar_type() == torch::kFloat32, "grad_rot_mats must be float32");
    TORCH_CHECK(rot_vecs.scalar_type() == torch::kFloat32, "rot_vecs must be float32");
    TORCH_CHECK(rot_mats_fwd.scalar_type() == torch::kFloat32, "rot_mats_fwd must be float32");

    TORCH_CHECK(grad_rot_mats.dim()==3 && grad_rot_mats.size(1)==3 && grad_rot_mats.size(2)==3);
    TORCH_CHECK(rot_vecs.dim()==2 && rot_vecs.size(1)==3);
    TORCH_CHECK(rot_mats_fwd.dim()==3 && rot_mats_fwd.size(1)==3 && rot_mats_fwd.size(2)==3);
    TORCH_CHECK(grad_rot_mats.size(0) == rot_vecs.size(0) && rot_vecs.size(0) == rot_mats_fwd.size(0));

    const int N = rot_vecs.size(0);
    const float epsilon = 1e-8f; // For theta_sq check

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rot_vecs.device());
    torch::Tensor grad_rot_vecs = torch::empty_like(rot_vecs, options);

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    torch::cuda::CUDAGuard device_guard(rot_vecs.device());

    batch_rodrigues_backward_kernel<<<blocks, threads_per_block>>>(
        grad_rot_mats.data_ptr<float>(),
        rot_vecs.data_ptr<float>(),
        rot_mats_fwd.data_ptr<float>(),
        grad_rot_vecs.data_ptr<float>(),
        N,
        epsilon
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_rodrigues_backward_kernel launch failed: ", cudaGetErrorString(err));

    return grad_rot_vecs;
}
