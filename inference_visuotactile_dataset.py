import os
import time
import warnings
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import lightning
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Import locali del progetto
from models.modules.ehm import EHM_v2 
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from utils.general_utils import ConfigDict, device_parser, add_extra_cfgs
from utils.get_video import images_to_video
from utils import rotation_converter as converter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")


TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER_SIZE = int(os.environ.get("PEAR_RENDER_SIZE", "512"))
OCCLUSION_START_FRAME = 50
DEFAULT_MYFUSION_PATH = Path.home() / "sensor_fusion_ws" / "fitting_ws"
MYFUSION_CACHE_ROOT = Path(os.environ.get("PEAR_MYFUSION_CACHE", "/tmp/pear_myfusion_cache"))
VisuotactileDatasetSource = None
myfusion_gender_from_name = None
prepared_files = None
make_visuotactile_metrics_row = None
write_visuotactile_metrics_csv = None


def setup_myfusion(myfusion_path):
    global VisuotactileDatasetSource, myfusion_gender_from_name, prepared_files
    global make_visuotactile_metrics_row, write_visuotactile_metrics_csv
    path = Path(myfusion_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MyFusion path does not exist: {path}")
    import_root = path.parent if path.name == "MyFusion" else path
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    from MyFusion.data_source.source_visuotactile_dataset import (
        VisuotactileDatasetSource as _VisuotactileDatasetSource,
        gender_from_name as _gender_from_name,
        make_visuotactile_metrics_row as _make_visuotactile_metrics_row,
        prepared_files as _prepared_files,
        write_visuotactile_metrics_csv as _write_visuotactile_metrics_csv,
    )
    VisuotactileDatasetSource = _VisuotactileDatasetSource
    myfusion_gender_from_name = _gender_from_name
    prepared_files = _prepared_files
    make_visuotactile_metrics_row = _make_visuotactile_metrics_row
    write_visuotactile_metrics_csv = _write_visuotactile_metrics_csv
    return import_root


def float_tensors(data):
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, dict):
        return {key: float_tensors(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(float_tensors(value) for value in data)
    return data
    

def build_cameras_kwargs(batch_size,focal_length):
    screen_size = torch.tensor(
        [RENDER_SIZE, RENDER_SIZE],
        device=TORCH_DEVICE,
    ).float()[None].repeat(batch_size, 1)
    cameras_kwargs = {
        'principal_point': torch.zeros(batch_size, 2, device=TORCH_DEVICE).float(), 
        'focal_length': focal_length, 
        'image_size': screen_size, 'device': TORCH_DEVICE,
    }
    return cameras_kwargs

def sequence_paths(input_path):
    return prepared_files(str(input_path))


def gender_hint_from_name(path):
    return myfusion_gender_from_name(Path(path), default="neutral")


def source_camera_count(source):
    data = source.data
    if "camera_names" in data.files:
        return int(np.asarray(data["camera_names"]).shape[0])
    if "camera_intrinsics_k" in data.files:
        intrinsics = np.asarray(data["camera_intrinsics_k"])
        if intrinsics.ndim >= 3:
            return int(intrinsics.shape[1])
        if intrinsics.ndim == 2:
            return int(intrinsics.shape[0])
    if getattr(source, "rgb", None) is not None:
        rgb = np.asarray(source.rgb)
        if rgb.ndim >= 5:
            return int(rgb.shape[1])
    return 2


def source_rgb_name(source, frame_idx):
    rgb_files = getattr(source, "rgb_files", None)
    if rgb_files is not None:
        return Path(rgb_files[frame_idx]).stem
    return f"{Path(source.path).stem}_frame{frame_idx:08d}"


def load_source_rgb_array(source, frame_idx):
    if getattr(source, "rgb_files", None) is not None:
        path = Path(source.rgb_files[frame_idx])
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is None:
                raise IOError(f"Fail to read {path}")
            return bgr[:, :, ::-1].astype(np.float32)
        loaded = np.load(path, mmap_mode="r")
        return np.asarray(loaded[loaded.files[0]] if hasattr(loaded, "files") else loaded)
    if getattr(source, "rgb", None) is None:
        raise FileNotFoundError(f"MyFusion source {source.path} has no RGB data")
    return np.asarray(source.rgb[frame_idx])


def source_camera_rgb(source, frame_idx, cam_idx, n_cams):
    rgb = load_source_rgb_array(source, frame_idx)
    if rgb.ndim == 4:
        return rgb[cam_idx].astype(np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame from MyFusion source, got shape {rgb.shape}")
    width = rgb.shape[1] // n_cams
    if width <= 0:
        raise ValueError(f"Cannot split RGB frame into {n_cams} horizontal camera views")
    return rgb[:, cam_idx * width:(cam_idx + 1) * width].astype(np.float32)


def apply_occlusion_texture(image, source, frame_idx, cam_idx, image_order="RGB"):
    try:
        occl_path = source._occlusion_mask_path(frame_idx, cam_idx)
    except AttributeError:
        root = Path(source.path) if Path(source.path).is_dir() else Path(source.path).parent
        occl_path = root / "occlusion_patches" / f"cam{int(cam_idx) + 1}" / f"frame_{int(frame_idx):08d}_texture.png"
    if not Path(occl_path).exists():
        return image
    occl_bgr = cv2.imread(str(occl_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if occl_bgr is None:
        return image
    occl_img = occl_bgr.astype(np.uint8) if image_order.upper() == "BGR" else occl_bgr[:, :, ::-1].astype(np.uint8)
    if occl_img.shape[:2] != image.shape[:2]:
        return image
    mask = np.any(occl_img != 0, axis=-1)
    if not mask.any():
        return image
    out = image.copy()
    out[mask] = occl_img[mask].astype(out.dtype)
    return out


def transform_pred_joints_to_base(pred, outputs, extrinsics, extrinsics_direction="camera_to_base", apply_pd_cam=False):
    if apply_pd_cam and outputs is not None and "pd_cam" in outputs:
        cam_rt = outputs["pd_cam"][0].detach().float().cpu().numpy().astype(np.float32)
        pred = pred @ cam_rt[:3, :3].T + cam_rt[:3, 3]

    if extrinsics is not None:
        ext = np.asarray(extrinsics, dtype=np.float32)
        if extrinsics_direction == "base_to_camera":
            rot = ext[:3, :3]
            trans = ext[:3, 3]
            pred = (pred - trans) @ rot
        else:
            pred = pred @ ext[:3, :3].T + ext[:3, 3]
    return pred


def transform_base_points_to_camera(points, extrinsics, extrinsics_direction="camera_to_base"):
    points = np.asarray(points, dtype=np.float32)
    if extrinsics is None:
        return points
    ext = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics_direction == "camera_to_base":
        return (points - ext[:3, 3]) @ ext[:3, :3]
    return points @ ext[:3, :3].T + ext[:3, 3]


def transform_pred_points_to_metric(pred, outputs, extrinsics, metric_frame="base", extrinsics_direction="camera_to_base", apply_pd_cam=False):
    pred = np.asarray(pred, dtype=np.float32)
    if metric_frame == "camera":
        if apply_pd_cam and outputs is not None and "pd_cam" in outputs:
            cam_rt = outputs["pd_cam"][0].detach().float().cpu().numpy().astype(np.float32)
            return pred @ cam_rt[:3, :3].T + cam_rt[:3, 3]
        return pred
    return transform_pred_joints_to_base(
        pred,
        outputs,
        extrinsics,
        extrinsics_direction=extrinsics_direction,
        apply_pd_cam=apply_pd_cam,
    )


def transform_gt_joints_to_metric(gt_joints, extrinsics, metric_frame="base", extrinsics_direction="camera_to_base"):
    if gt_joints is None or metric_frame == "base":
        return gt_joints
    gt = np.asarray(gt_joints, dtype=np.float32).copy()
    gt[..., :3] = transform_base_points_to_camera(gt[..., :3], extrinsics, extrinsics_direction=extrinsics_direction)
    return gt


def transform_markers_to_metric(markers, extrinsics, metric_frame="base", extrinsics_direction="camera_to_base"):
    if not markers or metric_frame == "base":
        return markers
    return {
        name: transform_base_points_to_camera(point[None], extrinsics, extrinsics_direction=extrinsics_direction)[0]
        for name, point in markers.items()
    }


def frame_info_for_metric(frame_info, metric_frame="base", extrinsics_direction="camera_to_base"):
    if metric_frame == "base":
        return frame_info
    info = dict(frame_info)
    info["gt_joints"] = transform_gt_joints_to_metric(
        frame_info.get("gt_joints"),
        frame_info.get("extrinsics"),
        metric_frame=metric_frame,
        extrinsics_direction=extrinsics_direction,
    )
    info["markers"] = transform_markers_to_metric(
        frame_info.get("markers") or {},
        frame_info.get("extrinsics"),
        metric_frame=metric_frame,
        extrinsics_direction=extrinsics_direction,
    )
    return info


def select_pred_gt_joints(pred, gt_joints, gt_indices):
    gt = np.asarray(gt_joints, dtype=np.float32)
    valid = np.isfinite(gt[..., :3]).all(axis=-1)
    if gt.shape[-1] > 3:
        valid &= gt[..., 3] > 0
    gt = gt[..., :3]

    if gt_indices is not None:
        indices = np.asarray(gt_indices, dtype=np.int64).reshape(-1)
        if indices.size == gt.shape[0] and indices.max(initial=-1) < pred.shape[0]:
            pred = pred[indices]
    if pred.shape[0] != gt.shape[0]:
        return None, None, None
    if not np.any(valid):
        return None, None, None
    return pred.astype(np.float32, copy=False), gt.astype(np.float32, copy=False), valid


def alignment_translation(pred, gt, valid, mode="root", root_index=0):
    if mode == "none":
        return np.zeros(3, dtype=np.float32)
    if mode == "mean":
        return (gt[valid].mean(axis=0) - pred[valid].mean(axis=0)).astype(np.float32)
    if root_index < 0 or root_index >= pred.shape[0] or not valid[root_index]:
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return np.zeros(3, dtype=np.float32)
        root_index = int(valid_indices[0])
    return (gt[root_index] - pred[root_index]).astype(np.float32)


def as_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def rotmat_tensor_to_axis_angle(value):
    if value is None:
        return None
    rotmats = value.detach().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value).float()
    rotmats = rotmats.reshape(-1, 3, 3)
    return converter.batch_matrix2axis(rotmats).detach().cpu().numpy().astype(np.float32).reshape(-1)


def first_rotmat(value):
    if value is None:
        return np.eye(3, dtype=np.float32)
    rotmats = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
    return rotmats.reshape(-1, 3, 3)[0].astype(np.float32)


def axis_angle_from_rotmat(rotmat):
    rotmat = torch.as_tensor(rotmat, dtype=torch.float32).reshape(1, 3, 3)
    return converter.batch_matrix2axis(rotmat)[0].detach().cpu().numpy().astype(np.float32)


def camera_rt_to_robot_base(cam_rot, cam_trans, extrinsics, extrinsics_direction="camera_to_base"):
    if extrinsics is None:
        return cam_rot.astype(np.float32), cam_trans.astype(np.float32)
    ext = np.asarray(extrinsics, dtype=np.float32)
    ext_rot = ext[:3, :3]
    ext_trans = ext[:3, 3]
    if extrinsics_direction == "base_to_camera":
        base_rot = ext_rot.T @ cam_rot
        base_trans = (cam_trans - ext_trans) @ ext_rot
    else:
        base_rot = ext_rot @ cam_rot
        base_trans = cam_trans @ ext_rot.T + ext_trans
    return base_rot.astype(np.float32), base_trans.astype(np.float32)


def pad_or_trim_vector(vector, size):
    out = np.zeros(size, dtype=np.float32)
    if vector is None:
        return out
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    out[:min(size, vector.size)] = vector[:size]
    return out


def estimated_smpl_params(outputs, extrinsics=None, extrinsics_direction="camera_to_base"):
    body_param = outputs.get("body_param", {})
    body_global_rot = first_rotmat(body_param.get("global_pose"))
    thetas = pad_or_trim_vector(rotmat_tensor_to_axis_angle(body_param.get("body_pose")), 69)

    cam_rot = np.eye(3, dtype=np.float32)
    cam_trans = np.zeros(3, dtype=np.float32)
    pd_cam = as_numpy(outputs.get("pd_cam"))
    if pd_cam is not None and pd_cam.ndim >= 3 and pd_cam.shape[-2] >= 3 and pd_cam.shape[-1] >= 4:
        cam_rt = pd_cam.reshape(-1, *pd_cam.shape[-2:])[0]
        cam_rot = cam_rt[:3, :3].astype(np.float32)
        cam_trans = cam_rt[:3, 3].astype(np.float32)

    base_cam_rot, trans = camera_rt_to_robot_base(
        cam_rot,
        cam_trans,
        extrinsics,
        extrinsics_direction=extrinsics_direction,
    )
    global_orientation = axis_angle_from_rotmat(base_cam_rot @ body_global_rot)
    full_pose = np.concatenate([global_orientation, thetas, trans]).astype(np.float32)
    betas = as_numpy(body_param.get("shape"))
    if betas is not None:
        betas = betas.reshape(betas.shape[0], -1)[0].astype(np.float32) if betas.ndim > 1 else betas.reshape(-1).astype(np.float32)
    return full_pose, betas


def create_open3d_visualizer():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Install open3d or run without --open3d_vis.") from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PEAR visuotactile prediction", width=1280, height=900)
    return o3d, vis


def open3d_marker_spheres(o3d, points, color, radius=0.02):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return None
    points = points.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        return None
    merged = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        sphere.translate(point.astype(np.float64))
        sphere.paint_uniform_color(color)
        merged += sphere
    merged.compute_vertex_normals()
    return merged


def update_open3d_visualizer(
    o3d,
    vis,
    meshes_vertices,
    faces,
    frame_info,
    wait_ms=1,
    reset_view=True,
):
    vis.clear_geometries()
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)]

    for vertices in meshes_vertices:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.78, 0.62, 0.48])
        geometries.append(mesh)

    gt_joints = frame_info.get("gt_joints")
    if gt_joints is not None:
        gt = np.asarray(gt_joints, dtype=np.float32)
        spheres = open3d_marker_spheres(o3d, gt[..., :3], [0.1, 0.85, 0.2], radius=0.018)
        if spheres is not None:
            geometries.append(spheres)

    markers = frame_info.get("markers") or {}
    if markers:
        marker_points = np.stack(list(markers.values()), axis=0)
        spheres = open3d_marker_spheres(o3d, marker_points, [1.0, 0.1, 0.05], radius=0.025)
        if spheres is not None:
            geometries.append(spheres)

    for geometry in geometries:
        vis.add_geometry(geometry, reset_bounding_box=False)
    if reset_view:
        vis.reset_view_point(True)

    vis.poll_events()
    vis.update_renderer()
    if wait_ms > 0:
        time.sleep(wait_ms / 1000.0)


def prepare_metric_joints(
    pd_smplx_dict,
    outputs,
    extrinsics,
    gt_joints,
    gt_indices,
    extrinsics_direction="camera_to_base",
    apply_pd_cam=False,
    metric_frame="camera",
    alignment="root",
    alignment_joint=0,
):
    if gt_joints is None:
        return None, None
    pred_joints = pd_smplx_dict.get("joints")
    if pred_joints is None:
        return None, None
    pred = pred_joints[0].detach().float().cpu().numpy().astype(np.float32)
    pred = transform_pred_points_to_metric(
        pred,
        outputs,
        extrinsics,
        metric_frame=metric_frame,
        extrinsics_direction=extrinsics_direction,
        apply_pd_cam=apply_pd_cam,
    )
    gt_joints = transform_gt_joints_to_metric(
        gt_joints,
        extrinsics,
        metric_frame=metric_frame,
        extrinsics_direction=extrinsics_direction,
    )
    pred, gt, valid = select_pred_gt_joints(pred, gt_joints, gt_indices)
    if pred is None:
        return None, None
    pred_mpjpe = pred + alignment_translation(pred, gt, valid, mode=alignment, root_index=alignment_joint)
    return pred_mpjpe[valid], gt[valid]


def align_vertices_for_visualization(
    vertices_base,
    pred_joints_base,
    gt_joints,
    gt_indices,
    alignment="root",
    alignment_joint=0,
):
    if gt_joints is None or alignment == "none":
        return vertices_base
    pred, gt, valid = select_pred_gt_joints(pred_joints_base, gt_joints, gt_indices)
    if pred is None:
        return vertices_base
    return vertices_base + alignment_translation(pred, gt, valid, mode=alignment, root_index=alignment_joint)


def camera_label(cam_idx):
    if cam_idx is None:
        return "cam_1"
    return f"cam_{int(cam_idx) + 1}"


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.6):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = bbox.astype(np.float32)
    return bbox

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def select_yolo_detections(result, max_detections=1):
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    if xyxy.shape[0] == 0:
        return xyxy
    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    heights = np.maximum(1e-6, xyxy[:, 3] - xyxy[:, 1])
    aspect_ratios = widths / heights
    order = np.lexsort((-widths, -aspect_ratios))
    xyxy = xyxy[order]
    if max_detections is None or max_detections <= 0 or xyxy.shape[0] <= max_detections:
        return xyxy
    return xyxy[:max_detections]


def draw_yolo_bboxes(image, bboxes, scale_factor=1.0, selected_idx=0):
    if image is None or bboxes is None or len(bboxes) == 0:
        return image

    img_h, img_w = image.shape[:2]
    inv_scale = 1.0 / scale_factor if scale_factor and scale_factor > 0 else 1.0
    thickness = max(2, int(round(min(img_h, img_w) / 350)))
    font_scale = max(0.45, min(img_h, img_w) / 900)

    for bbox_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = (np.asarray(bbox, dtype=np.float32) * inv_scale).tolist()
        x1 = int(np.clip(round(x1), 0, img_w - 1))
        y1 = int(np.clip(round(y1), 0, img_h - 1))
        x2 = int(np.clip(round(x2), 0, img_w - 1))
        y2 = int(np.clip(round(y2), 0, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        is_selected = bbox_idx == selected_idx
        color = (0, 255, 0) if is_selected else (0, 255, 255)
        box_thickness = thickness + 1 if is_selected else thickness
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        label = "selected" if is_selected else f"det_{bbox_idx}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y1 = max(0, y1 - text_h - baseline - 4)
        label_y2 = label_y1 + text_h + baseline + 4
        cv2.rectangle(image, (x1, label_y1), (min(img_w - 1, x1 + text_w + 6), label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return image


def inference(
    config_name="infer",
    devices="0",
    input_path=None,
    output_path=None,
    downscale=1.0,
    overlay_alpha=0.65,
    camera_index=1,
    start=0,
    end=-1,
    dump_outputs=True,
    camera_order="metadata",
    extrinsics_direction="camera_to_base",
    apply_pd_cam_to_metrics=False,
    metric_frame="camera",
    mpjpe_alignment="root",
    mpjpe_alignment_joint=0,
    open3d_vis=False,
    open3d_wait_ms=1,
    open3d_reset_view=False,
    max_detections=1,
    activate_occlusion=False,
    yolo_conf=0.1,
    yolo_iou=0.9,
    myfusion_path=DEFAULT_MYFUSION_PATH,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    myfusion_path = setup_myfusion(myfusion_path)
    print(f"[visuotactile] using MyFusion from {myfusion_path}")

    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)
    target_devices = device_parser(devices)
    init_iter = 1
    print(str(meta_cfg))
    print(
        f"[visuotactile] MPJPE transform: raw SMPL joints -> "
        f"{metric_frame} frame using {extrinsics_direction} extrinsics"
        f"{' after pd_cam' if apply_pd_cam_to_metrics else ''}"
        f", alignment={mpjpe_alignment}"
    )
    print(f"[visuotactile] RGB split camera order vs metadata: {camera_order}")

    body_renderer = None
    lights = None
    if dump_outputs:
        from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
        from pytorch3d.renderer import PointLights

        body_renderer = BodyRenderer("assets/SMPLX", RENDER_SIZE , focal_length=24.0 ).to(TORCH_DEVICE)
        body_renderer.eval()
        lights=PointLights(device=TORCH_DEVICE, location=[[0.0, -1.0, -10.0]])
        print(f"Using PyTorch3D renderer at {RENDER_SIZE}x{RENDER_SIZE}.")


    repo_id = "BestWJH/PEAR_models"  
    filename = "ehm_model_stage1.pt"  

    ehm_basemodel = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    ehm_model = Ehm_Pipeline(meta_cfg)
    _state=torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
    ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
    ehm_model.head.load_state_dict(_state['head'], strict=False)
    ehm_model = ehm_model.to(TORCH_DEVICE)
    ehm_model.eval()


    ehm = EHM_v2( "assets/FLAME", "assets/SMPLX")
    ehm = ehm.to(TORCH_DEVICE)
    ehm.eval()
    smplx_faces = ehm.smplx.faces_tensor.detach().cpu().numpy().astype(np.int32)

    o3d = None
    o3d_vis = None
    open3d_needs_initial_reset = True
    if open3d_vis:
        o3d, o3d_vis = create_open3d_visualizer()

    # init detector
    bbox_model = './model_zoo/yolov8x.pt'
    detector = YOLO(bbox_model)


    sequence_dirs = sequence_paths(input_path)
    if len(sequence_dirs) > 1:
        print(f"[visuotactile] found {len(sequence_dirs)} sequences under {input_path}")
    input_frames = []
    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name if sequence_dir.is_dir() else sequence_dir.stem
        gender_hint = gender_hint_from_name(sequence_dir)
        if gender_hint != "neutral":
            print(f"[visuotactile] sequence {sequence_name}: gender hint is {gender_hint}; using current PEAR/EHM model.")
        source = VisuotactileDatasetSource(
            path=sequence_dir,
            cache_root=MYFUSION_CACHE_ROOT,
            load_rgb=True,
            rebuild_cache=False,
            max_points=1,
            use_tactile=False,
            occlusion=False,
        )
        n_cams = source_camera_count(source)
        selected_cams = list(range(n_cams)) if camera_index < 0 else [int(camera_index)]
        if any(cam < 0 or cam >= n_cams for cam in selected_cams):
            raise ValueError(f"camera_index must be in [0,{n_cams - 1}] or -1, got {camera_index}")
        stop = len(source) if end is None or end < 0 or end > len(source) else end
        last_seen = None if start <= 0 else start - 1
        while True:
            try:
                source_frame, frame_idx = source.next_frame(last_seen=last_seen)
            except StopIteration:
                break
            if frame_idx >= stop:
                break
            last_seen = frame_idx
            frame_output_root = Path(output_path) / sequence_name if len(sequence_dirs) > 1 else Path(output_path)
            for cam_idx in selected_cams:
                meta_cam_idx = n_cams - 1 - cam_idx if camera_order == "reverse" else cam_idx
                input_frames.append({
                    "source": source,
                    "sequence": sequence_name,
                    "gender_hint": gender_hint,
                    "name": f"{source_rgb_name(source, frame_idx)}_cam{cam_idx}",
                    "frame_idx": frame_idx,
                    "timestamp_ns": source_frame.get("timestamp_ns", frame_idx),
                    "cam_idx": cam_idx,
                    "metadata_cam_idx": meta_cam_idx,
                    "n_cams": n_cams,
                    "extrinsics": source._retrieve_extrinsics(meta_cam_idx),
                    "gt_joints": source_frame.get("target_joints"),
                    "gt_indices": source_frame.get("target_joint_indices"),
                    "tactile_point": source_frame.get("tactile_point"),
                    "markers": source_frame.get("patch_markers") or {},
                    "output_dir": frame_output_root / camera_label(cam_idx).replace("_", ""),
                })
        print(
            f"[visuotactile] loaded {len(input_frames)} camera frames so far from MyFusion source {source.path} "
            f"({len(source)} frames, {n_cams} cameras, camera_order={camera_order})"
        )
    all_model_time = 0
    processed_images = 0
    processed_model_frames = 0
    successful_frames = 0
    frame_records = []
    source_metric_results = {}
    cached_pre_occlusion_bboxes = {}
    total_frames = len(input_frames)
    pbar = tqdm(input_frames, desc="Processing camera frames", unit="img")
    for idx, frame_info in enumerate(pbar):
        seq_name = frame_info.get("sequence", Path(input_path).name)
        img_name = frame_info["name"]
        cam_name = camera_label(frame_info["cam_idx"])
        frame_idx = int(frame_info["frame_idx"])
        bbox_cache_key = (seq_name, cam_name)
        frame_has_occlusion = activate_occlusion and frame_idx >= OCCLUSION_START_FRAME
        original_img = source_camera_rgb(
            frame_info["source"],
            frame_idx,
            int(frame_info["cam_idx"]),
            int(frame_info["n_cams"]),
        )
        if frame_has_occlusion:
            original_img = apply_occlusion_texture(
                original_img,
                frame_info["source"],
                frame_idx,
                int(frame_info["cam_idx"]),
                image_order="RGB",
            )
        original_img_height, original_img_width = original_img.shape[:2]

        # optionally downscale image for faster detection/ViT input
        if downscale is None or downscale <= 0 or downscale >= 1.0:
            scaled_img = original_img
            scale_factor = 1.0
        else:
            scale_factor = float(downscale)
            sw = max(1, int(original_img_width * scale_factor))
            sh = max(1, int(original_img_height * scale_factor))
            scaled_img = cv2.resize(original_img, (sw, sh), interpolation=cv2.INTER_LINEAR)

        if frame_has_occlusion:
            cached_bbox = cached_pre_occlusion_bboxes.get(bbox_cache_key)
            if cached_bbox is not None:
                all_yolo_bbox = cached_bbox.copy()
            else:
                all_yolo_bbox = np.empty((0, 4), dtype=np.float32)
                print(
                    f"[visuotactile] missing pre-occlusion bbox for {seq_name}/{cam_name} "
                    f"at frame {frame_idx}; skipping YOLO on occluded frame"
                )
        else:
            if detector is None:
                bbox_model = './model_zoo/yolov8x.pt'
                detector = YOLO(bbox_model)
            # detection on the (optional) scaled image; boxes are in scaled coordinates
            yolo_result = detector.predict(scaled_img,  # [h,w,3]  np
                                    device='cuda', 
                                    classes=0, 
                                    conf=yolo_conf,
                                    iou=yolo_iou,
                                    save=False, 
                                    verbose=False)[0]
            all_yolo_bbox = select_yolo_detections(yolo_result, max_detections=0)
        yolo_bbox = all_yolo_bbox[:1]
        if activate_occlusion and frame_idx < OCCLUSION_START_FRAME and len(yolo_bbox) > 0:
            cached_pre_occlusion_bboxes[bbox_cache_key] = np.asarray(yolo_bbox, dtype=np.float32).copy()
        
        vis_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_RGB2BGR) if dump_outputs else None

        if len(all_yolo_bbox) <1:
            if open3d_vis and o3d_vis is not None:
                display_frame_info = frame_info_for_metric(
                    frame_info,
                    metric_frame=metric_frame,
                    extrinsics_direction=extrinsics_direction,
                )
                update_open3d_visualizer(
                    o3d,
                    o3d_vis,
                    [],
                    smplx_faces,
                    display_frame_info,
                    wait_ms=open3d_wait_ms,
                    reset_view=False,
                )
            processed_images += 1
            mean_model_fps = processed_model_frames / all_model_time if all_model_time > 0 else 0.0
            frame_records.append({
                "sequence": seq_name,
                "gender_hint": frame_info.get("gender_hint", "neutral"),
                "frame_idx": frame_info["frame_idx"],
                "camera": cam_name,
                "metadata_camera": "" if frame_info["metadata_cam_idx"] is None else camera_label(frame_info["metadata_cam_idx"]),
                "image_name": img_name,
                "boxes": 0,
                "mpjpe_mm": "",
                "pa_mpjpe_mm": "",
                "model_time_s": "",
                "model_fps": "",
                "status": "no_person",
            })
            pbar.set_description(
                f"Processing camera frames success={successful_frames}/{total_frames} frame={processed_images}/{total_frames}"
            )
            pbar.set_postfix(
                frame=f"{processed_images}/{total_frames}",
                success=f"{successful_frames}/{total_frames}",
                seq=seq_name,
                # file=img_name,
                # cam=cam_name,
                # boxes=0,
                # model_fps=f"{mean_model_fps:.2f}",
                # mpjpe="n/a",
                # status="no person",
            )
            continue
        num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        frame_metric_rows = []
        frame_model_time = 0.0
        frame_meshes = []
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # map bbox from scaled coordinates back to original image coordinates
            if scale_factor != 1.0:
                inv_sf = 1.0 / scale_factor
                yolo_bbox_xywh = yolo_bbox_xywh * inv_sf

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=[256,256], 
                                ratio=1.6)          
            if bbox is None:
                continue
            # determine rotation for horizontal subjects: rotate upright
            rot = 0.0
            if bbox[2] > bbox[3] * 1.2:
                rot = 90.0
                  
            img_patch, trans, inv_trans = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.15, 
                                                rot=rot, 
                                                do_flip=False, 
                                                out_shape=[256,256])

            # normalize to [0,1] and move to device
            img_np = img_patch.astype(np.float32) / 255.0
            img_t = to_tensor(img_np, TORCH_DEVICE)  # H,W,C on device
            img_patch_t = img_t.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

            # run inference without autograd; building graphs here destroys FPS/memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_start = time.perf_counter()
            with torch.inference_mode():
                if TORCH_DEVICE.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = ehm_model(img_patch_t)
                else:
                    outputs = ehm_model(img_patch_t)

                outputs = float_tensors(outputs)
                pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')
                full_pose, betas = estimated_smpl_params(
                    outputs,
                    frame_info["extrinsics"],
                    extrinsics_direction=extrinsics_direction,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_model_time += time.perf_counter() - model_start

            with torch.inference_mode():
                pred_metric, gt_metric = prepare_metric_joints(
                    pd_smplx_dict,
                    outputs,
                    frame_info["extrinsics"],
                    frame_info["gt_joints"],
                    frame_info["gt_indices"],
                    extrinsics_direction=extrinsics_direction,
                    apply_pd_cam=apply_pd_cam_to_metrics,
                    metric_frame=metric_frame,
                    alignment=mpjpe_alignment,
                    alignment_joint=mpjpe_alignment_joint,
                )
                if pred_metric is not None:
                    frame_metric_rows.append(
                        make_visuotactile_metrics_row(
                            frame_idx=frame_idx,
                            timestamp_ns=frame_info.get("timestamp_ns", frame_idx),
                            pred_joints=pred_metric,
                            gt_joints=gt_metric,
                            tactile=frame_info.get("tactile_point"),
                            full_pose=full_pose,
                            betas=betas,
                        )
                    )

                if open3d_vis:
                    vertices = pd_smplx_dict['vertices'][0].detach().float().cpu().numpy().astype(np.float32)
                    pred_joints = pd_smplx_dict['joints'][0].detach().float().cpu().numpy().astype(np.float32)
                    vertices = transform_pred_points_to_metric(
                        vertices,
                        outputs,
                        frame_info["extrinsics"],
                        metric_frame=metric_frame,
                        extrinsics_direction=extrinsics_direction,
                        apply_pd_cam=apply_pd_cam_to_metrics,
                    )
                    pred_joints = transform_pred_points_to_metric(
                        pred_joints,
                        outputs,
                        frame_info["extrinsics"],
                        metric_frame=metric_frame,
                        extrinsics_direction=extrinsics_direction,
                        apply_pd_cam=apply_pd_cam_to_metrics,
                    )
                    display_frame_info = frame_info_for_metric(
                        frame_info,
                        metric_frame=metric_frame,
                        extrinsics_direction=extrinsics_direction,
                    )
                    vertices = align_vertices_for_visualization(
                        vertices,
                        pred_joints,
                        display_frame_info["gt_joints"],
                        display_frame_info["gt_indices"],
                        alignment=mpjpe_alignment,
                        alignment_joint=mpjpe_alignment_joint,
                    )
                    frame_meshes.append(vertices)
 

                if dump_outputs and body_renderer is not None and lights is not None:
                    pd_camera = GS_Camera(**build_cameras_kwargs(1,24), R = outputs['pd_cam'][0:0+1,:3,:3], T = outputs['pd_cam'][0:0+1,:3,3])
                    pd_mesh_rgba = body_renderer.render_mesh(pd_smplx_dict['vertices'][None, 0,...], pd_camera, lights=lights )

            if dump_outputs and vis_img is not None:
                pd_mesh_rgba = (pd_mesh_rgba.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)

                pd_mesh_img = cv2.cvtColor(pd_mesh_rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
                pd_mesh_alpha = pd_mesh_rgba[:, :, 3] if pd_mesh_rgba.shape[-1] > 3 else np.any(pd_mesh_img > 0, axis=-1).astype(np.uint8) * 255

                pd_mesh_img = cv2.resize(pd_mesh_img, ( 256, 256 ), interpolation=cv2.INTER_AREA)
                pd_mesh_alpha = cv2.resize(pd_mesh_alpha, ( 256, 256 ), interpolation=cv2.INTER_AREA)

                H, W = original_img.shape[:2]

                mesh_on_orig = cv2.warpAffine(
                    pd_mesh_img,
                    inv_trans,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                alpha_on_orig = cv2.warpAffine(
                    pd_mesh_alpha,
                    inv_trans,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                mask = alpha_on_orig > 10

                vis_img[mask] = (
                    overlay_alpha * mesh_on_orig[mask].astype(np.float32)
                    + (1.0 - overlay_alpha) * vis_img[mask].astype(np.float32)
                ).astype(np.uint8)

        if open3d_vis and o3d_vis is not None:
            display_frame_info = frame_info_for_metric(
                frame_info,
                metric_frame=metric_frame,
                extrinsics_direction=extrinsics_direction,
            )
            update_open3d_visualizer(
                o3d,
                o3d_vis,
                frame_meshes,
                smplx_faces,
                display_frame_info,
                wait_ms=open3d_wait_ms,
                reset_view=open3d_needs_initial_reset and len(frame_meshes) > 0,
            )
            if frame_meshes:
                open3d_needs_initial_reset = False

        finite_metric_rows = [
            row for row in frame_metric_rows
            if np.isfinite(float(row.get("mpjpe_mm", float("nan"))))
        ]
        frame_metric_row = min(finite_metric_rows, key=lambda row: float(row["mpjpe_mm"])) if finite_metric_rows else None
        frame_mpjpe = None if frame_metric_row is None else float(frame_metric_row["mpjpe_mm"])
        frame_pa_mpjpe = None if frame_metric_row is None else float(frame_metric_row["pa_mpjpe_mm"])
        if frame_metric_row is not None:
            result_key = (seq_name, cam_name)
            result = source_metric_results.setdefault(
                result_key,
                {"sequence": f"{seq_name}_{cam_name}", "rows": []},
            )
            result["rows"].append(frame_metric_row)
        if dump_outputs and vis_img is not None:
            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
            if frame_has_occlusion:
                vis_img = apply_occlusion_texture(
                    vis_img,
                    frame_info["source"],
                    frame_idx,
                    int(frame_info["cam_idx"]),
                    image_order="BGR",
                )
            draw_yolo_bboxes(vis_img, all_yolo_bbox, scale_factor=scale_factor, selected_idx=0)
            frame_output_dir = Path(frame_info.get("output_dir", output_path))
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_output_dir / f"mesh_{img_name}.jpg"), vis_img)
        frame_model_fps = 1.0 / frame_model_time if frame_model_time > 0 else 0.0
        all_model_time += frame_model_time
        processed_images += 1
        processed_model_frames += 1
        successful_frames += 1
        mean_model_fps = processed_model_frames / all_model_time if all_model_time > 0 else 0.0
        frame_records.append({
            "sequence": seq_name,
            "gender_hint": frame_info.get("gender_hint", "neutral"),
            "frame_idx": frame_info["frame_idx"],
            "camera": cam_name,
            "metadata_camera": "" if frame_info["metadata_cam_idx"] is None else camera_label(frame_info["metadata_cam_idx"]),
            "image_name": img_name,
            "boxes": num_bbox,
            "mpjpe_mm": f"{frame_mpjpe:.4f}" if frame_mpjpe is not None else "",
            "pa_mpjpe_mm": f"{frame_pa_mpjpe:.4f}" if frame_pa_mpjpe is not None else "",
            "model_time_s": f"{frame_model_time:.6f}",
            "model_fps": f"{frame_model_fps:.4f}",
            "status": "saved" if dump_outputs else "processed",
        })
        pbar.set_description(
            f"Processing camera frames success={successful_frames}/{total_frames} frame={processed_images}/{total_frames}"
        )
        pbar.set_postfix(
            frame=f"{processed_images}/{total_frames}",
            success=f"{successful_frames}/{total_frames}",
            seq=seq_name,
            file=img_name,
            cam=cam_name,
            boxes=num_bbox,
            meshes=len(frame_meshes),
            model_fps=f"{mean_model_fps:.2f}",
            mpjpe=f"{frame_mpjpe:.1f}mm" if frame_mpjpe is not None else "n/a",
            pa=f"{frame_pa_mpjpe:.1f}mm" if frame_pa_mpjpe is not None else "n/a",
            status="saved" if dump_outputs else "processed",
        )

    # print mean inference FPS (detection+model+render per image)
    if all_model_time > 0 and processed_model_frames > 0:
        mean_fps = processed_model_frames / all_model_time
    else:
        mean_fps = 0.0
    print(f"Processed {processed_images} frames. Model-only inference FPS: {mean_fps:.2f}")
    for result in source_metric_results.values():
        write_visuotactile_metrics_csv(result, output_path)

    if dump_outputs:
        video_dirs = sorted({str(row.get("output_dir", output_path)) for row in input_frames})
        for video_dir in video_dirs:
            images_to_video(
                video_dir,
                os.path.join(video_dir, "video.mp4"),
                fps=30
            )
    if o3d_vis is not None:
        o3d_vis.destroy_window()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../Nardi/m1_2026-04-28-13-05-40_0', type=str)
    parser.add_argument('--output_path', default='outputs', type=str)
    parser.add_argument('--end', default=-1, type=int, help='Exclusive dataset RGB frame index to process. Use -1 for all remaining frames.')
    parser.add_argument('--open3d_vis', action='store_true', help='Show predicted SMPL-X meshes, target joints, and markers in Open3D frame by frame.')
    parser.add_argument('--camera-index', dest='camera_index', default=1, type=int, help='Camera view to process after horizontal RGB split.')
    parser.add_argument('--activate-occlusion', action='store_true', help='Apply per-frame occlusion textures from the occlusion_patches folder.')
    parser.add_argument('--no-dump-outputs', action='store_true', help='Do not save overlay images or videos; metrics CSV are still saved.')
    parser.add_argument('--yolo-conf', dest='yolo_conf', default=0.4, type=float, help='YOLO person detection confidence threshold.')
    parser.add_argument('--yolo-iou', dest='yolo_iou', default=0.9, type=float, help='YOLO NMS IoU threshold. Higher values keep more overlapping boxes.')
    parser.add_argument(
        '--myfusion-path',
        default=str(DEFAULT_MYFUSION_PATH),
        type=str,
        help='Path to the MyFusion package folder or to the workspace folder that contains it.',
    )
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    torch.set_float32_matmul_precision('high')
    inference(
        input_path=args.input_path,
        output_path=args.output_path,
        camera_index=args.camera_index,
        end=args.end,
        dump_outputs=not args.no_dump_outputs,
        open3d_vis=args.open3d_vis,
        activate_occlusion=args.activate_occlusion,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        myfusion_path=args.myfusion_path,
        max_detections=2
    )
