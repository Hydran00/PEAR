import os
import shutil
import site
import sys
import importlib

def migrate_precompiled_packages():
    target_site_packages = site.getsitepackages()[0]

    packages_to_check = {
        'pytorch3d': ['pytorch3d', 'pytorch3d-0.7.8.dist-info'],
        'chumpy': ['chumpy', 'chumpy-0.70.dist-info']
    }
    
    print(f"📦 Starting migration check in: {os.getcwd()}")
    
    for pkg_name, folders in packages_to_check.items():
        # 检查包是否已经安装且可用
        try:
            importlib.import_module(pkg_name)
            print(f"✅ {pkg_name} is already available. Skipping.")
            continue
        except ImportError:
            print(f"🔍 {pkg_name} not found. Preparing to migrate...")

        for folder in folders:
            src = os.path.abspath(folder)
            dst = os.path.join(target_site_packages, folder)
            
            if os.path.exists(src):
                try:
                    if os.path.exists(dst):
                        print(f"⚠️ Removing existing {dst}...")
                        shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                    
                    print(f"🚚 Copying {folder} to site-packages...")
                    shutil.copytree(src, dst)
                except Exception as e:
                    print(f"❌ Failed to migrate {folder}: {e}")
            else:
                print(f"❓ Source {folder} missing in root directory.")

    # 2. 核心补丁：刷新搜索路径并置顶
    importlib.invalidate_caches()
    if target_site_packages not in sys.path:
        sys.path.insert(0, target_site_packages)

    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        os.environ["LD_LIBRARY_PATH"] = torch_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        print(f"🔗 LD_LIBRARY_PATH updated with torch libs.")
    except Exception as e:
        print(f"⚠️ Failed to set LD_LIBRARY_PATH: {e}")


# migrate_precompiled_packages()

try:
    import chumpy
    import pytorch3d
    from pytorch3d import _C
    print(f"🎉 All systems go! PyTorch3D GPU: {hasattr(_C, 'rasterize_meshes')}")
except Exception as e:
    print(f"🚨 Validation failed: {e}")


import gradio as gr
import os
import json
import numpy as np
import cv2
import base64
import time
import shutil
import glob
from pathlib import Path
from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import atexit
import uuid
import logging
from huggingface_hub import hf_hub_download
import torch
import decord
try:
    import spaces
except ImportError:
    # Fallback for local development
    def space(func):
        return func
import imageio
from tqdm import tqdm

from einops import rearrange

from models.modules.ehm import EHM_v2
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
from utils.general_utils import (
    ConfigDict,
    rtqdm,
    device_parser,
    add_extra_cfgs,
)
from scipy.signal import savgol_filter
try:
    import spaces  # for HuggingFace Spaces
except ImportError:
    # Fallback for local development
    def spaces(func):
        return func


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Device & runtime configuration (CPU / GPU agnostic)
# -------------------------------------------------------------------------
TORCH_DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER_SIZE = int(os.environ.get("PEAR_RENDER_SIZE", "512"))
ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
if TORCH_DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True



meta_cfg = ConfigDict(
    model_config_path=os.path.join("configs", "infer.yaml")
)
meta_cfg = add_extra_cfgs(meta_cfg)

# Renderer / model init (on selected device)
body_renderer = BodyRenderer("assets/SMPLX", RENDER_SIZE, focal_length=24.0)


repo_id = "BestWJH/PEAR_models"  
filename = "ehm_model_stage1.pt"  

ehm_basemodel = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
ehm_model = Ehm_Pipeline(meta_cfg)

_state = torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
ehm_model.head.load_state_dict(_state['head'], strict=False)

ehm = EHM_v2("assets/FLAME", "assets/SMPLX")


# lights = PointLights(device=TORCH_DEVICE, location=[[0.0, -1.0, -10.0]])

# 2. 将 lights 的初始化移入函数，或确保它能动态转换
def get_lights(device):
    return PointLights(device=device, location=[[0.0, -1.0, -10.0]])




# Thread pool for delayed deletion
thread_pool_executor = ThreadPoolExecutor(max_workers=2)

def build_cameras_kwargs(batch_size, focal_length):
    screen_size = (
        torch.tensor([RENDER_SIZE, RENDER_SIZE], device=TORCH_DEVICE)
        .float()[None]
        .repeat(batch_size, 1)
    )
    cameras_kwargs = {
        "principal_point": torch.zeros(batch_size, 2, device=TORCH_DEVICE).float(),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": TORCH_DEVICE,
    }
    return cameras_kwargs



def pad_and_resize(img, target_size=512):

    h, w = img.shape[:2]

    # 等比例缩放
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建黑色背景图像并粘贴到中心
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_img

def delete_later(path: Union[str, os.PathLike], delay: int = 600):
    """Delete file or directory after specified delay (default 10 minutes)"""
    def _delete():
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
    
    def _wait_and_delete():
        time.sleep(delay)
        _delete()
    
    thread_pool_executor.submit(_wait_and_delete)
    atexit.register(_delete)


def create_user_temp_dir():
    """Create a unique temporary directory for each user session"""
    session_id = str(uuid.uuid4())[:8]  # Short unique ID
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(
        PROJECT_ROOT, "temp_local", f"session_{session_id}"
    )

    os.makedirs(temp_dir, exist_ok=True)
    
    # Schedule deletion after 10 minutes
    delete_later(temp_dir, delay=600)
    
    return temp_dir

def get_video_name(video_path):
    """Extract video name without extension"""
    return os.path.splitext(os.path.basename(video_path))[0]

def extract_first_frame(video_path):
    """Extract first frame from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return None
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return None

def numpy_to_base64(arr):
    """Convert numpy array to base64 string"""
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def polynomial_smooth(sequence, window_size=5, polyorder=2):

    seq = sequence.detach().float().cpu().numpy()
    if seq.ndim < 2:
        raise ValueError(f"输入必须至少是 2 维，当前 shape={seq.shape}")

    if window_size % 2 == 0:
        raise ValueError("window_size 必须是奇数")
    if polyorder >= window_size:
        raise ValueError("polyorder 必须小于 window_size")

    # Savitzky–Golay 沿着 axis=0 (时间维) 平滑
    smoothed = savgol_filter(seq, window_length=window_size, polyorder=polyorder, axis=0, mode='interp')
    return smoothed.astype(np.float32, copy=False)


def smoothed_tensor(sequence, window_size=5, polyorder=2):
    return torch.as_tensor(
        polynomial_smooth(sequence, window_size=window_size, polyorder=polyorder),
        dtype=torch.float32,
        device=TORCH_DEVICE,
    )


class Open3DOffscreenMeshRenderer:
    def __init__(self, faces, image_size=512):
        import open3d as o3d

        self.o3d = o3d
        self.image_size = int(image_size)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.image_size,
            self.image_size,
        )
        self.renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.material.base_color = [0.86, 0.70, 0.58, 1.0]

    def render(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        mesh = self.o3d.geometry.TriangleMesh(
            self.o3d.utility.Vector3dVector(vertices),
            self.o3d.utility.Vector3iVector(self.faces),
        )
        mesh.compute_vertex_normals()

        self.renderer.scene.clear_geometry()
        self.renderer.scene.add_geometry("body", mesh, self.material)

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center().astype(np.float32)
        extent = float(np.max(bbox.get_extent()))
        distance = max(extent * 2.4, 1.0)
        eye = center + np.array([0.0, 0.0, distance], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.renderer.setup_camera(35.0, center, eye, up)

        image = np.asarray(self.renderer.render_to_image())
        if image.ndim == 3 and image.shape[-1] > 3:
            image = image[..., :3]
        return image.astype(np.uint8, copy=False)


def build_demo_renderer(faces):
    backend = os.environ.get("PEAR_RENDER_BACKEND", "pytorch3d").lower()
    if backend != "open3d":
        print(f"🎨 Using PyTorch3D renderer at {RENDER_SIZE}x{RENDER_SIZE}.")
        return None

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    force_open3d = os.environ.get("PEAR_FORCE_OPEN3D", "0") == "1"
    if not has_display and not force_open3d:
        print("⚠️ Open3D offscreen skipped: no DISPLAY/WAYLAND_DISPLAY. Set PEAR_FORCE_OPEN3D=1 to try anyway.")
        print(f"🎨 Using PyTorch3D renderer at {RENDER_SIZE}x{RENDER_SIZE}.")
        return None

    try:
        image_size = int(os.environ.get("PEAR_OPEN3D_RENDER_SIZE", "512"))
        renderer = Open3DOffscreenMeshRenderer(faces, image_size=image_size)
        print(f"🎨 Using Open3D offscreen renderer at {image_size}x{image_size}.")
        return renderer
    except Exception as e:
        print(f"⚠️ Open3D renderer unavailable ({e}); falling back to PyTorch3D.")
        return None


def handle_video_upload(video):
    """Handle video upload and extract first frame"""
    if video is None:
        return None
    
    # Create user-specific temporary directory
    user_temp_dir = create_user_temp_dir()
    
    # Get original video name and copy to temp directory
    # if isinstance(video, str):
    #     video_name = get_video_name(video)
    #     video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
    #     shutil.copy(video, video_path)
    # else:
    #     video_name = get_video_name(video.name)
    #     video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
    #     with open(video_path, 'wb') as f:
    #         f.write(video.read())

    # 确定输入源路径/
    input_source = video if isinstance(video, str) else video.name

    video_name = get_video_name(input_source)
    video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")

    try:
        # 使用 imageio 读取视频
        reader = imageio.get_reader(input_source)
        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 30)
        max_frames = int(fps * 3)
        writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8)
        
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            writer.append_data(frame)
        
        reader.close()
        writer.close()
        print(f"成功截取前 3 秒视频并保存至: {video_path}")

    except Exception as e:
        print(f"imageio 处理视频失败: {e}")


    print(f"📁 Video saved to: {video_path}")
    
    # Extract first frame
    frame = extract_first_frame(video_path)
    if frame is None:
        return None
    
    # Resize frame to have minimum side length of 336
    h, w = frame.shape[:2]
    scale = 336 / min(h, w)
    new_h, new_w = int(h * scale)//2*2, int(w * scale)//2*2
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Store frame data with temp directory info
    frame_data = {
        'data': numpy_to_base64(frame),
        'shape': frame.shape,
        'dtype': str(frame.dtype),
        'temp_dir': user_temp_dir,
        'video_name': video_name,
        'video_path': video_path
    }
    
    # Get video-specific settings
    print(f"🎬 Video path: '{video}' -> Video name: '{video_name}'")

    return (json.dumps(frame_data)       )

@spaces.GPU
@torch.no_grad()
def mesh_inference(temp_dir, video_name):

    global body_renderer, ehm_model, ehm
    body_renderer = body_renderer.to(TORCH_DEVICE)
    ehm_model = ehm_model.to(TORCH_DEVICE)
    ehm = ehm.to(TORCH_DEVICE)
    body_renderer.eval()
    ehm_model.eval()
    ehm.eval()
    lights = get_lights(TORCH_DEVICE)

    # ... 后续推理逻辑 ...
    # Setup paths
    video_path = os.path.join(temp_dir, f"{video_name}.mp4")

    out_dir = os.path.join(temp_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load video using decord
    video_reader = decord.VideoReader(video_path)
    # Don't load all frames at once; process frame-by-frame to save memory.


    # Run the EHM pipeline
    print(f"🎯 Running EHM pipeline...")
    ehm_out_dir = os.path.join(out_dir, "ehm_results")
    os.makedirs(ehm_out_dir, exist_ok=True)


    
    all_meshes_img = []
    vertices_list = []
    body_sequence = []
    flame_sequence = []
    cam_sequence = []
    # Process each frame with EHM
    frame_pbar = tqdm(range(len(video_reader)), desc="EHM frames", unit="frame")
    for i in frame_pbar:
        frame = video_reader[i].asnumpy()
        # TODO: Apply EHM processing to frame
        resized = pad_and_resize(frame, target_size=256)
        # convert to float tensor on device and normalize
        img_np = resized.astype(np.float32) / 255.0
        img_t = to_tensor(img_np, TORCH_DEVICE)  # H,W,C on device
        img_patch = img_t.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

        # run model with autocast for faster fp16 inference on CUDA
        if TORCH_DEVICE.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = ehm_model(img_patch)
        else:
            outputs = ehm_model(img_patch)
        frame_pbar.set_postfix(frame=i + 1)


        body_sequence.append(outputs['body_param'])
        flame_sequence.append(outputs['flame_param'])
        cam_sequence.append(outputs['pd_cam'])


    fields1 = [
        "global_pose", "body_pose", "left_hand_pose", "right_hand_pose",
        "hand_scale", "head_scale", "exp", "shape"
    ]

    processed1 = {}
    for key in fields1:
        data_list = [seq[key] for seq in body_sequence]
        data_tensor = torch.cat(data_list, dim=0)
        processed1[key] = smoothed_tensor(data_tensor, window_size=7, polyorder=2)

    global_pose = processed1["global_pose"]
    body_pose = processed1["body_pose"]
    left_hand_pose = processed1["left_hand_pose"]
    right_hand_pose = processed1["right_hand_pose"]
    hand_scale = processed1["hand_scale"]
    head_scale = processed1["head_scale"]
    exp = processed1["exp"]
    shape = processed1["shape"]

    # 第二步：从 eye_pose_params 中提取字段并平滑
    fields2 = [
        "eye_pose_params", "pose_params", "jaw_params",
        "eyelid_params", "expression_params", "shape_params"
    ]

    processed2 = {}
    for key in fields2:
        data_list = [seq[key] for seq in flame_sequence]  # 这里我猜你原意是从 eye_pose_params 取
        data_tensor = torch.cat(data_list, dim=0)
        processed2[key] = smoothed_tensor(data_tensor, window_size=5, polyorder=2)

    eye_pose_params = processed2["eye_pose_params"]
    pose_params = processed2["pose_params"]
    jaw_params = processed2["jaw_params"]
    eyelid_params = processed2["eyelid_params"]
    expression_params = processed2["expression_params"]
    shape_params = processed2["shape_params"]


    cam_sequence = torch.cat(cam_sequence, dim=0)
    cam_sequence = smoothed_tensor(cam_sequence, window_size=7, polyorder=2)



    faces = body_renderer.faces[0].detach().cpu().numpy()
    open3d_renderer = build_demo_renderer(faces)
    render_pbar = tqdm(range(global_pose.shape[0]), desc="Rendering mesh", unit="frame")
    for idx in render_pbar:
        
        pd_cam = cam_sequence[idx:idx+1]

        body_dict = {
            "global_pose": global_pose[idx:idx+1],
            "body_pose" : body_pose[idx:idx+1], 
            "left_hand_pose" : left_hand_pose[idx:idx+1], 
            "right_hand_pose" : right_hand_pose[idx:idx+1],
            "hand_scale" : hand_scale[idx:idx+1], 
            "head_scale" : head_scale[idx:idx+1], 
            "exp" :  exp[idx:idx+1]  ,
            "shape": shape[idx:idx+1],
            'eye_pose' :None,
            'jaw_pose' : None,
            'joints_offset' : None
        }
        flame_dict ={
            "eye_pose_params" : eye_pose_params[idx:idx+1], 
            "pose_params" : pose_params[idx:idx+1],
            "jaw_params" : jaw_params[idx:idx+1], 
            "eyelid_params" : eyelid_params[idx:idx+1], 
            "expression_params" :  expression_params[idx:idx+1]  ,
            "shape_params": shape_params[idx:idx+1]
        }

        pd_smplx_dict = ehm(body_dict, flame_dict, pose_type='aa')
        vertices_np = pd_smplx_dict['vertices'][0].detach().float().cpu().numpy()
        vertices_list.append(vertices_np)

        if open3d_renderer is not None:
            pd_mesh_img = open3d_renderer.render(vertices_np)
        else:
            pd_camera = GS_Camera(**build_cameras_kwargs(1,24), R = pd_cam[0:1,:3,:3], T = pd_cam[0:1,:3,3])
            pd_mesh_img = body_renderer.render_mesh(pd_smplx_dict['vertices'][None, 0,...], pd_camera, lights=lights,)
            pd_mesh_img = (pd_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
            # pd_mesh_img = cv2.cvtColor(pd_mesh_img.copy(), cv2.COLOR_RGB2BGR)
        all_meshes_img.append(pd_mesh_img)
        render_pbar.set_postfix(
            backend="open3d" if open3d_renderer is not None else "pytorch3d",
            frame=idx + 1,
        )



    # Save results
    mesh_video_path = os.path.join(out_dir, "mesh_video.mp4")
    # Write a browser-compatible MP4:
    # - H.264 + yuv420p for broad HTML5 support
    # - faststart so it can stream/play immediately
    if len(all_meshes_img) == 0:
        raise RuntimeError("No frames generated for mesh video.")
    fps = 30
    try:
        writer = imageio.get_writer(
            mesh_video_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "faststart"],
            macro_block_size=None,
        )
        for img in all_meshes_img:
            # Ensure even H/W for yuv420p
            h, w = img.shape[:2]
            img2 = img[: h - (h % 2), : w - (w % 2)]
            writer.append_data(img2)
        writer.close()
    except Exception as e:
        print(f"⚠️ imageio ffmpeg writer failed ({e}); falling back to mimwrite")
        imageio.mimwrite(mesh_video_path, all_meshes_img, fps=fps)

    # Save a portable npz (avoid storing Trimesh objects which will fail).
    # vertices: (T, V, 3), faces: (F, 3)
    vertices = np.stack(vertices_list, axis=0) if len(vertices_list) > 0 else np.empty((0, 0, 3), dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "results.npz"), vertices=vertices, faces=faces)


    print(f"✅ EHM processing completed.")



# @torch.no_grad()
def launch_viz(original_image_state):
    """Launch visualization with user-specific temp directory"""
    if original_image_state is None:
        return None, None

    try:
        # Get user's temp directory from stored frame data
        frame_data = json.loads(original_image_state)

        temp_dir = frame_data.get("temp_dir", "temp_local")
        video_name = frame_data.get("video_name", "video")

        print(f"🚀 Starting recover for video: {video_name}")

        # Run tracker
        print("🎯 Running...")
        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True)

        mesh_inference(temp_dir, video_name)

        delete_later(temp_dir, delay=600)

        npz_path = os.path.join(out_dir, "results.npz")
        mesh_video = os.path.join(out_dir, "mesh_video.mp4")

        if os.path.exists(mesh_video):
            print("✅ Tracking completed successfully!")
            # Returning the path lets gr.Video handle file serving & controls
            return mesh_video, (npz_path if os.path.exists(npz_path) else None)
        else:
            print("❌ Tracking failed - no results generated")
            return None, None

    except Exception as e:
        print(f"❌ Error in launch_viz: {e}")
        return None, None



# Create the Gradio interface
print("🎨 Creating Gradio interface...")

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🎯 [PEAR](https://wujh2001.github.io/PEAR/)",
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .gr-button {
        margin: 5px;
    }
    .gr-form {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    /* 移除 gr.Group 的默认灰色背景 */
    .gr-form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* 固定3D可视化器尺寸（纯白卡片样式） */
    #viz_container {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 12px !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.6) !important;
        background: #ffffff !important;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    /* 固定“左侧上传”视频组件高度（不要影响右侧结果视频） */
    #video_input {
        height: 300px !important;
        min-height: 300px !important;
        max-height: 300px !important;
    }
    #video_input video {
        height: 260px !important;
        max-height: 260px !important;
        object-fit: contain !important;
        background: #f8f9fa;
    }
    #video_input .gr-video-player {
        height: 260px !important;
        max-height: 260px !important;
    }

    /* 强力移除examples的灰色背景 - 使用更通用的选择器 */
    .horizontal-examples,
    .horizontal-examples > *,
    .horizontal-examples * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Examples组件水平滚动样式 */
    .horizontal-examples [data-testid="examples"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] > div {
        background: transparent !important;
        background-color: transparent !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin;
        scrollbar-color: #667eea transparent;
        padding: 0 !important;
        margin-top: 10px;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] table {
        display: flex !important;
        flex-wrap: nowrap !important;
        min-width: max-content !important;
        gap: 15px !important;
        padding: 10px 0;
        background: transparent !important;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] tbody {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 15px !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] tr {
        display: flex !important;
        flex-direction: column !important;
        min-width: 160px !important;
        max-width: 160px !important;
        margin: 0 !important;
        background: white !important;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
        cursor: pointer;
        overflow: hidden;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] tr:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.25);
    }
    
    .horizontal-examples [data-testid="examples"] td {
        text-align: center !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:first-child {
        padding: 0 !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] video {
        border-radius: 8px 8px 0 0 !important;
        width: 100% !important;
        height: 90px !important;
        object-fit: cover !important;
        background: #f8f9fa !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:last-child {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #333 !important;
        padding: 8px 12px !important;
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%) !important;
        border-radius: 0 0 8px 8px;
    }
    
    /* 滚动条样式 */
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar {
        height: 8px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 4px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    """
) as demo:
    
    # Add prominent main title
    
    gr.Markdown("""
    # ✨ PEAR
                
    Welcome to [PEAR](https://wujh2001.github.io/PEAR/)! This interface allows you to human mesh from a single image.
    Please cite our paper and give us a star 🌟 if you find this project useful!
    
    **⚡ Quick Start:** Upload video → Click "Start Tracking Now!"
    
    **🔬 Advanced Usage with PEAR:**
    
    """)
    
    # Status indicator
    status_indicator = gr.Markdown("**Reminder:** 🟢 This app currently only supports single human-centered video inputs (3 seconds). For other input types, you can customize the app manually.")
    
    # Main content area - video upload left, 3D visualization right
    with gr.Row():
        with gr.Column(scale=1):
            # Video upload section
            gr.Markdown("### 📂 Select Video")
            
            # Define video_input here so it can be referenced in examples
            video_input = gr.Video(
                label="Upload Video or Select Example",
                format="mp4",
                height=250,  # Matched height with 3D viz
                elem_id="video_input"
            )
                

            # Traditional examples but with horizontal scroll styling
            gr.Markdown("🎨**Examples:** (scroll horizontally to see all videos)")
            with gr.Row(elem_classes=["horizontal-examples"]):
                # Horizontal video examples with slider
                # gr.HTML("<div style='margin-top: 5px;'></div>")
                gr.Examples(
                    examples=[
                        ["example/example_1.mp4"],
                        ["example/example_2.mp4"],
                    ],
                    inputs=[video_input],
                    outputs=[video_input],
                    fn=None,
                    cache_examples=False,
                    label="",
                    examples_per_page=6  # Show 6 examples per page so they can wrap to multiple rows
                )
        
        with gr.Column(scale=2):
            # 3D Visualization - wider and taller to match left side
            with gr.Group():
                gr.Markdown("### ✨ Human Mesh Visualization")
                # Output-only video: no upload controls, only play the result
                viz_video = gr.Video(
                    label="Reconstructed Human Mesh Video",
                    height=650,
                    elem_id="viz_container",
                    interactive=False,
                    autoplay=False,
                    sources=None,  # hide upload / webcam buttons
                )

    # Start button section - below video area
    with gr.Row():
        with gr.Column(scale=3):
            launch_btn = gr.Button("🚀 Start Tracking Now!", variant="primary", size="lg")
        with gr.Column(scale=1):
            clear_all_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")

    # Downloads section - visible download area
    with gr.Row():
        with gr.Column():
            parameters_download = gr.File(
                label="📄 Download Mesh Results (vertices and faces)",
                interactive=False,
            )


    # GitHub Star Section
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #e8eaff 0%, #f0f2ff 100%); 
                border-radius: 8px; padding: 20px; margin: 15px 0; 
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.15);'>
        <div style='text-align: center;'>
            <h3 style='color: #4a5568; margin: 0 0 10px 0; font-size: 18px; font-weight: 600;'>
                ⭐ Love PEAR? Give us a Star! ⭐
            </h3>
            <p style='color: #666; margin: 0 0 15px 0; font-size: 14px; line-height: 1.5;'>
                Help us grow by starring our repository on GitHub! Your support means a lot to the community. 🚀
            </p>
            <a href="https://wujh2001.github.io/PEAR/" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(102, 126, 234, 0.1); color: #4a5568; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(102, 126, 234, 0.2);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(102, 126, 234, 0.15)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(102, 126, 234, 0.1)'; this.style.transform='translateY(0)'">
                <span style='font-size: 16px;'>⭐</span>
                Star PEAR on GitHub
            </a>
        </div>
    </div>
    """)

    
    # Footer
    gr.HTML("""
    <div style='text-align: center; margin: 20px 0 10px 0;'>
        <span style='font-size: 12px; color: #888; font-style: italic;'>
            Powered by PEAR | Built with ❤️ for the Computer Vision Community
        </span>
    </div>
    """)


    # Hidden state variables
    original_image_state = gr.State(None)



    # # Event handlers
    video_input.change(
        fn=handle_video_upload,
        inputs=[video_input],
        outputs=[original_image_state],
        api_name=False # ⭐ 关键
    )

    launch_btn.click(
    fn=launch_viz,
        inputs=[original_image_state],
        outputs=[viz_video, parameters_download],
        api_name=False # ⭐ 关键
    )


# Launch the interface
if __name__ == "__main__":
    print("🌟 Launching PEAR Local Version...")
    print("🔗 Running in Local Processing Mode")
    demo.queue().launch()
    
