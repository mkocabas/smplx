import os
import torch
import numpy as np
import smplx # For loading model parameters and faces
# cuda_smplx_ops should be importable after compilation
# import cuda_smplx_ops 

# Visualization library
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not found. Visualization will be skipped or use Matplotlib if implemented.")
    # Optional: Matplotlib fallback
    # try:
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     MATPLOTLIB_AVAILABLE = True
    # except ImportError:
    #     MATPLOTLIB_AVAILABLE = False
    #     print("Matplotlib for 3D scatter not found. No visualization possible.")
    MATPLOTLIB_AVAILABLE = False # Keep it simple for now

# Model configuration (consistent with tests)
MODEL_PATH_HERE = os.environ.get('SMPLX_TEST_MODELS_PATH', 'data/smplx_models')
MODEL_TYPE = 'smplx'
GENDER = 'neutral'
NUM_BETAS = 10
NUM_EXPRESSION_COEFFS = 10
NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS = 15
NUM_JAW_JOINTS = 1
DTYPE = torch.float32 # For model operations

def get_sample_inputs_viz(batch_size, device='cuda'):
    # Similar to test_forward get_sample_inputs, but uses DTYPE
    betas = torch.randn(batch_size, NUM_BETAS, dtype=DTYPE, device=device) * 0.5 # Smaller variation
    expression = torch.randn(batch_size, NUM_EXPRESSION_COEFFS, dtype=DTYPE, device=device) * 0.5
    
    global_orient = torch.zeros(batch_size, 3, dtype=DTYPE, device=device) 
    # Example: A-pose like, slight arm rotation
    body_pose_np = np.zeros((batch_size, NUM_BODY_JOINTS * 3), dtype=np.float32)
    # body_pose_np[:, 3*13 + 1] = 0.6 # Right shoulder abduct
    # body_pose_np[:, 3*14 + 1] = -0.6 # Left shoulder abduct
    body_pose = torch.tensor(body_pose_np, dtype=DTYPE, device=device)
    
    jaw_pose = torch.zeros(batch_size, NUM_JAW_JOINTS * 3, dtype=DTYPE, device=device)
    left_hand_pose = torch.zeros(batch_size, NUM_HAND_JOINTS * 3, dtype=DTYPE, device=device)
    right_hand_pose = torch.zeros(batch_size, NUM_HAND_JOINTS * 3, dtype=DTYPE, device=device)
    
    return betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose

def visualize_mesh_open3d(vertices_np, faces_np):
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for visualization.")
        return
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    # Optional: Set view control, lighting
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.8)
    # ctr.rotate(x=100,y=100) # Example view adjustment
    print("Displaying mesh with Open3D. Close the window to continue.")
    vis.run()
    vis.destroy_window()

def main_visualization():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run CUDA SMPL-X demo.")
        return
    device = torch.device("cuda:0")

    try:
        import cuda_smplx_ops
    except ImportError:
        print("CUDA extension 'cuda_smplx_ops' not found. Please compile it first.")
        print("Try: python setup.py install (from project root)")
        return

    # 1. Load reference SMPL-X model to get parameters (especially faces)
    try:
        # Load with float32 for parameters, as CUDA ops expect this.
        ref_model_params = smplx.create(
            MODEL_PATH_HERE, model_type=MODEL_TYPE, gender=GENDER, use_pca=False,
            num_betas=NUM_BETAS, num_expression_coeffs=NUM_EXPRESSION_COEFFS, ext='npz'
        ).to(dtype=DTYPE) 
    except Exception as e:
        print(f"Could not load reference SMPL-X model parameters from {MODEL_PATH_HERE}. Error: {e}")
        print("Skipping visualization.")
        return

    faces_np = ref_model_params.faces.astype(np.int32) # For Open3D

    # 2. Prepare model constants for the CUDA function
    v_template_const = ref_model_params.v_template.detach().clone().to(device=device, dtype=DTYPE)
    combined_shapedirs_const = torch.cat(
        [ref_model_params.shapedirs, ref_model_params.exprdirs], dim=2
    ).detach().clone().to(device=device, dtype=DTYPE)
    
    num_model_vertices = ref_model_params.get_num_verts()
    posedirs_ref = ref_model_params.posedirs.reshape(num_model_vertices * 3, -1)
    posedirs_const = posedirs_ref.transpose(0,1).contiguous().detach().clone().to(device=device, dtype=DTYPE)
    
    J_regressor_const = ref_model_params.J_regressor.detach().clone().to(device=device, dtype=DTYPE)
    parents_const = ref_model_params.parents.detach().clone().to(device=device, dtype=torch.int64)
    lbs_weights_const = ref_model_params.lbs_weights.detach().clone().to(device=device, dtype=DTYPE)
    
    use_pca_hands_const = False
    lh_components_const = torch.empty(0, device=device, dtype=DTYPE)
    rh_components_const = torch::empty(0, device=device, dtype=DTYPE)

    # 3. Get sample inputs
    batch_size = 1 # Visualize one mesh at a time
    betas, expression, global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose = \
        get_sample_inputs_viz(batch_size, device=device)

    # 4. Execute CUDA forward pass (using the autograd-enabled function)
    outputs_cuda = cuda_smplx_ops.smplx_cuda_traced(
        betas, expression, global_orient, body_pose, 
        left_hand_pose, right_hand_pose, jaw_pose, 
        v_template_const, combined_shapedirs_const, posedirs_const,
        J_regressor_const, parents_const, lbs_weights_const,
        use_pca_hands_const, lh_components_const, rh_components_const
    )
    output_vertices_cuda = outputs_cuda[0]
    
    # 5. Visualize
    vertices_np = output_vertices_cuda[0].detach().cpu().numpy()
    
    print(f"Generated {vertices_np.shape[0]} vertices.")
    if OPEN3D_AVAILABLE:
        visualize_mesh_open3d(vertices_np, faces_np)
    elif MATPLOTLIB_AVAILABLE:
        print("Matplotlib fallback not implemented in this script.")
        pass
    else:
        print("No visualization library (Open3D or Matplotlib 3D) available.")
        print("Vertices mean:", vertices_np.mean(axis=0))

if __name__ == "__main__":
    print(f"Looking for SMPL-X models in: {os.path.abspath(MODEL_PATH_HERE)}")
    if not os.path.exists(MODEL_PATH_HERE):
        print(f"Warning: Model path {MODEL_PATH_HERE} does not exist.")
    main_visualization()
