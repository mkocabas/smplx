import os
import torch
import numpy as np
import smplx # Assuming this library is available for ground truth
# import cuda_smplx_ops # This will be the compiled CUDA extension

# Placeholder for where SMPL-X model files are stored.
# In a real CI, this path would point to actual model files.
# For local testing, users need to download models from SMPL-X website
# and place them appropriately.
MODEL_PATH_HERE = os.environ.get('SMPLX_TEST_MODELS_PATH', 'data/smplx_models') 
# Example: export SMPLX_TEST_MODELS_PATH=/path/to/your/smplx/models

# Configuration
MODEL_TYPE = 'smplx'
GENDER = 'neutral' # or 'male', 'female'
NUM_BETAS = 10 # Standard for SMPL-X shape
NUM_EXPRESSION_COEFFS = 10 # Standard for SMPL-X expression
NUM_BODY_JOINTS = 21 # SMPL-X body joints (excluding global, jaw, hands, eyes)
NUM_HAND_JOINTS = 15 # Per hand
NUM_JAW_JOINTS = 1 # Jaw joint for SMPLX

# Tolerance for float comparisons
TOLERANCE = 1e-5

def get_sample_inputs(batch_size, device='cuda'):
    betas = torch.randn(batch_size, NUM_BETAS, dtype=torch.float32, device=device)
    expression = torch.randn(batch_size, NUM_EXPRESSION_COEFFS, dtype=torch.float32, device=device)
    
    global_orient = torch.randn(batch_size, 3, dtype=torch.float32, device=device) * 0.1
    body_pose = torch.randn(batch_size, NUM_BODY_JOINTS * 3, dtype=torch.float32, device=device) * 0.1
    jaw_pose = torch.randn(batch_size, NUM_JAW_JOINTS * 3, dtype=torch.float32, device=device) * 0.1 # 1 jaw joint * 3
    
    # For non-PCA hands, pose is axis-angle (NUM_HAND_JOINTS * 3)
    left_hand_pose = torch.randn(batch_size, NUM_HAND_JOINTS * 3, dtype=torch.float32, device=device) * 0.1
    right_hand_pose = torch.randn(batch_size, NUM_HAND_JOINTS * 3, dtype=torch.float32, device=device) * 0.1
    
    return betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose

def test_smplx_forward_pass():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return
    device = torch.device("cuda:0")

    # Try to import the compiled CUDA extension
    try:
        import cuda_smplx_ops 
    except ImportError:
        print("CUDA extension 'cuda_smplx_ops' not found. Compile first. Skipping test.")
        return

    batch_size = 2

    # 1. Load reference SMPL-X model
    try:
        ref_model = smplx.create(
            MODEL_PATH_HERE,
            model_type=MODEL_TYPE,
            gender=GENDER,
            use_pca=False, # Using full axis-angle for hands for this test
            num_betas=NUM_BETAS,
            num_expression_coeffs=NUM_EXPRESSION_COEFFS,
            ext='npz' # or 'pkl' depending on your model files
        ).to(device)
        ref_model.eval()
    except Exception as e:
        print(f"Could not load reference SMPL-X model from {MODEL_PATH_HERE}. Error: {e}")
        print("Please ensure SMPL-X models are downloaded and MODEL_PATH_HERE is set correctly.")
        print("Skipping test.")
        return

    # 2. Prepare sample inputs
    betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose = \
        get_sample_inputs(batch_size, device=device)

    # 3. Execute reference model forward pass
    with torch.no_grad():
        ref_output = ref_model(
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            return_verts=True,
            return_full_pose=False 
        )
        ref_vertices = ref_output.vertices
        ref_joints = ref_output.joints 

    # 4. Prepare inputs for CUDA version
    v_template_cuda = ref_model.v_template.detach().clone().to(device=device, dtype=torch.float32)
    
    combined_shapedirs = torch.cat([ref_model.shapedirs, ref_model.exprdirs], dim=2).detach().clone().to(device=device, dtype=torch.float32)
    
    num_model_vertices = ref_model.get_num_verts()
        
    posedirs_ref = ref_model.posedirs.reshape(num_model_vertices * 3, -1) 
    posedirs_cuda = posedirs_ref.transpose(0,1).contiguous().detach().clone().to(device=device, dtype=torch.float32)

    J_regressor_cuda = ref_model.J_regressor.detach().clone().to(device=device, dtype=torch.float32) 
    
    parents_cuda = ref_model.parents.detach().clone().to(device=device, dtype=torch.int64)
    
    lbs_weights_cuda = ref_model.lbs_weights.detach().clone().to(device=device, dtype=torch.float32)

    use_pca_hands_cuda = False 
    lh_components_cuda = torch.empty(0, device=device, dtype=torch.float32)
    rh_components_cuda = torch.empty(0, device=device, dtype=torch.float32)

    # 5. Execute CUDA forward pass (no_grad version first)
    cuda_outputs = cuda_smplx_ops.smplx_forward_cuda_no_grad(
        betas, expression,
        global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose,
        v_template_cuda, combined_shapedirs, posedirs_cuda, J_regressor_cuda,
        parents_cuda, lbs_weights_cuda,
        use_pca_hands_cuda, lh_components_cuda, rh_components_cuda
    )
    cuda_vertices = cuda_outputs[0]
    cuda_joints = cuda_outputs[1] 

    # 6. Compare outputs
    assert torch.allclose(cuda_vertices, ref_vertices, atol=TOLERANCE), \
        f"CUDA vertices differ from reference. Max diff: {torch.max(torch.abs(cuda_vertices - ref_vertices))}"
    print("Forward pass vertices match reference.")

    # The number of joints from smplx model output might include extra landmarks.
    # CUDA version outputs num_model_joints (kinematic chain).
    num_kinematic_joints = cuda_joints.shape[1] 
    
    assert cuda_joints.shape[0] == ref_joints.shape[0] and \
           cuda_joints.shape[2] == ref_joints.shape[2], \
           f"Batch size or dim mismatch: CUDA joints {cuda_joints.shape}, Ref joints {ref_joints.shape}"
    
    assert ref_joints.shape[1] >= num_kinematic_joints, \
           f"Reference model has fewer joints ({ref_joints.shape[1]}) than CUDA kinematic model ({num_kinematic_joints})."

    assert torch.allclose(cuda_joints, ref_joints[:, :num_kinematic_joints, :], atol=TOLERANCE), \
        f"CUDA joints differ from reference. Max diff: {torch.max(torch.abs(cuda_joints - ref_joints[:, :num_kinematic_joints, :]))}"
    print("Forward pass joints match reference.")

    print("test_smplx_forward_pass PASSED.")

if __name__ == "__main__":
    print(f"Looking for SMPL-X models in: {os.path.abspath(MODEL_PATH_HERE)}")
    if not os.path.exists(MODEL_PATH_HERE):
        print(f"Warning: Model path {MODEL_PATH_HERE} does not exist or is not accessible.")
        print("Please set SMPLX_TEST_MODELS_PATH environment variable or place models in 'data/smplx_models'.")
        print("Attempting to run test anyway, but it will likely fail at model loading.")
    
    test_smplx_forward_pass()
