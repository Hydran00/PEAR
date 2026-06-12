from models.modules.ehm import EHM_v2 
from models.pipeline.ehm_pipeline import Ehm_Pipeline
import os
import torch
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
import csv
import cv2
import time
import warnings
from pathlib import Path

# suppress ultralytics torch.load FutureWarning about weights_only (coming from external lib)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from ultralytics import YOLO
import os
import torch
import argparse
import lightning
import numpy as np
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict, device_parser, add_extra_cfgs
)
from huggingface_hub import hf_hub_download
from tqdm import tqdm


from utils.get_video import images_to_video

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER_SIZE = int(os.environ.get("PEAR_RENDER_SIZE", "512"))
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
RGB_EXTENSIONS = IMAGE_EXTENSIONS + ('.npy', '.npz')


def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    iou = intersection_area / union_area
    return iou


def non_max_suppression(bboxes, iou_threshold):
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    selected_bboxes = []
    while len(bboxes) > 0:
        current_bbox = bboxes[0]
        selected_bboxes.append(current_bbox)
        bboxes = bboxes[1:]
        
        remaining_bboxes = []
        for bbox in bboxes:
            iou = calculate_iou(current_bbox, bbox)
            if iou < iou_threshold:
                remaining_bboxes.append(bbox)
                
        bboxes = remaining_bboxes
        
    return selected_bboxes

def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]

    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_img


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

def load_img(path, order='RGB', scale=1):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    if scale != 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC  
        )

    img = img.astype(np.float32)
    return img


def _metadata_scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def _dataset_camera_count(metadata):
    if "camera_names" in metadata.files:
        return int(np.asarray(metadata["camera_names"]).shape[0])
    if "camera_intrinsics_k" in metadata.files:
        intrinsics = np.asarray(metadata["camera_intrinsics_k"])
        if intrinsics.ndim >= 3:
            return int(intrinsics.shape[1])
        if intrinsics.ndim == 2:
            return int(intrinsics.shape[0])
    return 2


def _metadata_paths(root, metadata, key, fallback_dir):
    field = f"{key}_files"
    if field in metadata.files:
        return [root / str(rel) for rel in metadata[field]]

    folder = root / fallback_dir
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in RGB_EXTENSIONS
    )


def _frame_intrinsics(metadata, frame_idx, cam_idx):
    if "camera_intrinsics_k" not in metadata.files:
        return None
    intrinsics = np.asarray(metadata["camera_intrinsics_k"])
    if intrinsics.ndim == 3:
        return intrinsics[frame_idx, cam_idx].astype(np.float32)
    if intrinsics.ndim == 2:
        return intrinsics[cam_idx].astype(np.float32)
    return None


def _frame_extrinsics(metadata, frame_idx, cam_idx):
    if "camera_extrinsics" not in metadata.files:
        return None
    extrinsics = np.asarray(metadata["camera_extrinsics"])
    if extrinsics.ndim == 4:
        return extrinsics[frame_idx, cam_idx].astype(np.float32)
    if extrinsics.ndim == 3:
        return extrinsics[cam_idx].astype(np.float32)
    return None


def metadata_camera_index(cam_idx, n_cams, camera_order):
    if camera_order == "reverse":
        return n_cams - 1 - cam_idx
    return cam_idx


def sequence_paths(input_path):
    root = Path(input_path)
    if root.is_dir() and (root / "metadata.npz").exists():
        return [root]
    if root.is_dir():
        paths = sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.npz").exists())
        if paths:
            return paths
    return [root]


def gender_hint_from_name(path):
    name = Path(path).name.lower()
    if name.startswith("m"):
        return "male"
    if name.startswith("f"):
        return "female"
    return "neutral"


def _frame_gt(metadata, frame_idx):
    if "target_joints" not in metadata.files:
        return None, None
    joints = np.asarray(metadata["target_joints"][frame_idx], dtype=np.float32)
    indices = None
    if "target_joint_indices" in metadata.files:
        indices = np.asarray(metadata["target_joint_indices"], dtype=np.int64)
        if indices.ndim > 1 and indices.shape[0] == metadata["target_joints"].shape[0]:
            indices = indices[frame_idx]
    return joints, indices


def _frame_markers(metadata, frame_idx):
    markers = {}
    if "belly_markers" in metadata.files:
        marker_points = np.asarray(metadata["belly_markers"][frame_idx], dtype=np.float32)
        marker_names = metadata["belly_marker_names"] if "belly_marker_names" in metadata.files else None
        for marker_idx, point in enumerate(marker_points):
            if not np.isfinite(point).all():
                continue
            if marker_names is None:
                name = f"marker_{marker_idx}"
            else:
                raw_name = marker_names[marker_idx]
                name = raw_name.decode("utf-8") if isinstance(raw_name, (bytes, np.bytes_)) else str(raw_name)
            markers[name] = point.astype(np.float32)
    return markers


def build_input_frames(input_path, camera_index=-1, start=0, end=-1, camera_order="metadata"):
    root = Path(input_path)
    metadata_path = root / "metadata.npz"
    if metadata_path.exists() and root.is_dir():
        metadata = np.load(metadata_path, allow_pickle=False)
        rgb_paths = _metadata_paths(root, metadata, "rgb", "rgb")
        if not rgb_paths:
            raise FileNotFoundError(f"No RGB frames found in {root / 'rgb'} and no rgb_files in metadata.npz")

        n_cams = _dataset_camera_count(metadata)
        selected_cams = list(range(n_cams)) if camera_index < 0 else [int(camera_index)]
        if any(cam < 0 or cam >= n_cams for cam in selected_cams):
            raise ValueError(f"camera_index must be in [0,{n_cams - 1}] or -1, got {camera_index}")

        total = len(rgb_paths)
        stop = total if end is None or end < 0 or end > total else end
        frames = []
        for frame_idx in range(max(0, start), stop):
            for cam_idx in selected_cams:
                meta_cam_idx = metadata_camera_index(cam_idx, n_cams, camera_order)
                gt_joints, gt_indices = _frame_gt(metadata, frame_idx)
                frames.append({
                    "path": rgb_paths[frame_idx],
                    "name": f"{rgb_paths[frame_idx].stem}_cam{cam_idx}",
                    "frame_idx": frame_idx,
                    "cam_idx": cam_idx,
                    "metadata_cam_idx": meta_cam_idx,
                    "n_cams": n_cams,
                    "metadata": metadata,
                    "intrinsics": _frame_intrinsics(metadata, frame_idx, meta_cam_idx),
                    "extrinsics": _frame_extrinsics(metadata, frame_idx, meta_cam_idx),
                    "gt_joints": gt_joints,
                    "gt_indices": gt_indices,
                    "markers": _frame_markers(metadata, frame_idx),
                    "split_horizontal": True,
                })
        print(
            f"[visuotactile] loaded {len(frames)} camera frames from {root} "
            f"({total} rgb frames, {n_cams} cameras, camera_order={camera_order})"
        )
        if "camera_intrinsics_k" in metadata.files:
            print("[visuotactile] camera intrinsics loaded from metadata.npz")
        if "camera_extrinsics" in metadata.files:
            print("[visuotactile] camera extrinsics loaded from metadata.npz")
        if "target_joints" in metadata.files:
            print("[visuotactile] ground truth target_joints loaded from metadata.npz")
        return frames

    image_paths = sorted(
        root / f for f in os.listdir(root)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    total = len(image_paths)
    stop = total if end is None or end < 0 or end > total else end
    return [
        {
            "path": image_paths[idx],
            "name": image_paths[idx].stem,
            "frame_idx": idx,
            "cam_idx": None,
            "metadata_cam_idx": None,
            "n_cams": 1,
            "metadata": None,
            "intrinsics": None,
            "extrinsics": None,
            "gt_joints": None,
            "gt_indices": None,
            "markers": {},
            "split_horizontal": False,
        }
        for idx in range(max(0, start), stop)
    ]


def load_input_frame(frame_info):
    if not frame_info["split_horizontal"]:
        return load_img(str(frame_info["path"]), scale=1)

    path = Path(frame_info["path"])
    n_cams = int(frame_info["n_cams"])
    cam_idx = int(frame_info["cam_idx"])
    if path.suffix.lower() in (".npy", ".npz"):
        loaded = np.load(path, mmap_mode="r")
        rgb = loaded[loaded.files[0]] if hasattr(loaded, "files") else loaded
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            return rgb[cam_idx].astype(np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB array in {path}, got shape {rgb.shape}")
        width = rgb.shape[1] // n_cams
        if width <= 0:
            raise ValueError(f"Cannot split {path} into {n_cams} horizontal camera views")
        return rgb[:, cam_idx * width:(cam_idx + 1) * width].astype(np.float32)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise IOError(f"Fail to read {path}")
    width = bgr.shape[1] // n_cams
    if width <= 0:
        raise ValueError(f"Cannot split {path} into {n_cams} horizontal camera views")
    bgr = bgr[:, cam_idx * width:(cam_idx + 1) * width]
    return bgr[:, :, ::-1].astype(np.float32)


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


def umeyama_alignment(source, target):
    src = source.astype(np.float64)
    dst = target.astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = (src_centered ** 2).sum() / src_centered.shape[0]
    if variance <= 0:
        return source.astype(np.float32)
    covariance = (dst_centered.T @ src_centered) / src_centered.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    scale = np.trace(np.diag(singular_values)) / variance
    translation = dst_mean - scale * (rotation @ src_mean)
    return (scale * (src @ rotation.T) + translation).astype(np.float32)


def create_open3d_visualizer():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Install open3d or run without --open3d_vis.") from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PEAR visuotactile prediction", width=1280, height=900)
    return o3d, vis


def open3d_point_cloud(o3d, points, color):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return None
    points = points.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    return cloud


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


def target_joint_metrics_mm(
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
        return None
    pred_joints = pd_smplx_dict.get("joints")
    if pred_joints is None:
        return None
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
    mpjpe = float(np.linalg.norm(pred_mpjpe[valid] - gt[valid], axis=-1).mean() * 1000.0)
    pred_pa = umeyama_alignment(pred[valid], gt[valid])
    pa_mpjpe = float(np.linalg.norm(pred_pa - gt[valid], axis=-1).mean() * 1000.0)
    return mpjpe, pa_mpjpe


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


def write_metrics_csv(output_path, frame_records, summary_records):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_csv = output_dir / "visuotactile_frame_metrics.csv"
    with frame_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "gender_hint",
                "frame_idx",
                "camera",
                "metadata_camera",
                "image_name",
                "boxes",
                "mpjpe_mm",
                "pa_mpjpe_mm",
                "model_time_s",
                "model_fps",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_records)

    summary_csv = output_dir / "visuotactile_summary_metrics.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "camera",
                "frames",
                "frames_with_model",
                "frames_with_mpjpe",
                "mean_mpjpe_mm",
                "mean_pa_mpjpe_mm",
                "mean_model_fps",
                "overall_model_fps",
                "total_model_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_records)

    return frame_csv, summary_csv


def summarize_metrics(frame_records):
    groups = sorted({(row.get("sequence", ""), row["camera"]) for row in frame_records})
    summary = []
    for sequence, camera in groups:
        rows = [row for row in frame_records if row.get("sequence", "") == sequence and row["camera"] == camera]
        times = [float(row["model_time_s"]) for row in rows if row["model_time_s"] != ""]
        fps_values = [float(row["model_fps"]) for row in rows if row["model_fps"] != ""]
        mpjpes = [float(row["mpjpe_mm"]) for row in rows if row["mpjpe_mm"] != ""]
        pa_mpjpes = [float(row["pa_mpjpe_mm"]) for row in rows if row["pa_mpjpe_mm"] != ""]
        total_time = float(np.sum(times)) if times else 0.0
        summary.append({
            "sequence": sequence,
            "camera": camera,
            "frames": len(rows),
            "frames_with_model": len(times),
            "frames_with_mpjpe": len(mpjpes),
            "mean_mpjpe_mm": f"{float(np.mean(mpjpes)):.4f}" if mpjpes else "",
            "mean_pa_mpjpe_mm": f"{float(np.mean(pa_mpjpes)):.4f}" if pa_mpjpes else "",
            "mean_model_fps": f"{float(np.mean(fps_values)):.4f}" if fps_values else "",
            "overall_model_fps": f"{(len(times) / total_time):.4f}" if total_time > 0 else "",
            "total_model_time_s": f"{total_time:.6f}",
        })
    return summary


def overall_metrics(frame_records):
    times = [float(row["model_time_s"]) for row in frame_records if row["model_time_s"] != ""]
    fps_values = [float(row["model_fps"]) for row in frame_records if row["model_fps"] != ""]
    mpjpes = [float(row["mpjpe_mm"]) for row in frame_records if row["mpjpe_mm"] != ""]
    pa_mpjpes = [float(row["pa_mpjpe_mm"]) for row in frame_records if row["pa_mpjpe_mm"] != ""]
    total_time = float(np.sum(times)) if times else 0.0
    return {
        "sequence": "ALL",
        "camera": "ALL",
        "frames": len(frame_records),
        "frames_with_model": len(times),
        "frames_with_mpjpe": len(mpjpes),
        "mean_mpjpe_mm": f"{float(np.mean(mpjpes)):.4f}" if mpjpes else "",
        "mean_pa_mpjpe_mm": f"{float(np.mean(pa_mpjpes)):.4f}" if pa_mpjpes else "",
        "mean_model_fps": f"{float(np.mean(fps_values)):.4f}" if fps_values else "",
        "overall_model_fps": f"{(len(times) / total_time):.4f}" if total_time > 0 else "",
        "total_model_time_s": f"{total_time:.6f}",
    }

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


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
    if max_detections is None or max_detections <= 0 or xyxy.shape[0] <= max_detections:
        return xyxy
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
    if conf is not None:
        order = np.argsort(-conf)
    else:
        area = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-area)
    return xyxy[order[:max_detections]]


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
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
        sequence_name = sequence_dir.name
        gender_hint = gender_hint_from_name(sequence_dir)
        if gender_hint != "neutral":
            print(f"[visuotactile] sequence {sequence_name}: gender hint is {gender_hint}; using current PEAR/EHM model.")
        frames = build_input_frames(
            sequence_dir,
            camera_index=camera_index,
            start=start,
            end=end,
            camera_order=camera_order,
        )
        for frame in frames:
            frame["sequence"] = sequence_name
            frame["gender_hint"] = gender_hint
            frame["output_dir"] = Path(output_path) / sequence_name if len(sequence_dirs) > 1 else Path(output_path)
        input_frames.extend(frames)
    all_model_time = 0
    processed_images = 0
    processed_model_frames = 0
    gt_errors_mm = []
    frame_records = []
    # transform = transforms.ToTensor()  # not used; we'll normalize explicitly
    pbar = tqdm(input_frames, desc="Processing camera frames", unit="img")
    for idx, frame_info in enumerate(pbar):
        seq_name = frame_info.get("sequence", Path(input_path).name)
        img_name = frame_info["name"]
        cam_name = camera_label(frame_info["cam_idx"])
        # start timing this frame's inference (includes detection + model + render)
        t0 = time.time()
        original_img = load_input_frame(frame_info)
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

        # detection on the (optional) scaled image; boxes are in scaled coordinates
        yolo_result = detector.predict(scaled_img,  # [h,w,3]  np
                                device='cuda', 
                                classes=0, 
                                conf=0.3, 
                                save=False, 
                                verbose=False)[0]
        yolo_bbox = select_yolo_detections(yolo_result, max_detections=max_detections)
        
        vis_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_RGB2BGR) if dump_outputs else None

        if len(yolo_bbox) <1:
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
            pbar.set_postfix(
                seq=seq_name,
                file=img_name,
                cam=cam_name,
                boxes=0,
                model_fps=f"{mean_model_fps:.2f}",
                mpjpe="n/a",
                status="no person",
            )
            continue
        num_bbox = len(yolo_bbox)


        # loop all detected bboxes
        frame_metric_pairs = []
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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_model_time += time.perf_counter() - model_start

            with torch.inference_mode():
                mpjpe_mm, pa_mpjpe_mm = target_joint_metrics_mm(
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
                if mpjpe_mm is not None:
                    frame_metric_pairs.append((mpjpe_mm, pa_mpjpe_mm))

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

        frame_mpjpe, frame_pa_mpjpe = min(frame_metric_pairs, key=lambda item: item[0]) if frame_metric_pairs else (None, None)
        if frame_metric_pairs:
            gt_errors_mm.append(frame_mpjpe)
        if dump_outputs and vis_img is not None:
            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
            frame_output_dir = Path(frame_info.get("output_dir", output_path))
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_output_dir / f"mesh_{img_name}.jpg"), vis_img)
        frame_model_fps = 1.0 / frame_model_time if frame_model_time > 0 else 0.0
        all_model_time += frame_model_time
        processed_images += 1
        processed_model_frames += 1
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
        valid_cam_errors = [
            float(row["mpjpe_mm"])
            for row in frame_records
            if row["camera"] == cam_name and row["mpjpe_mm"] != ""
        ]
        valid_cam_pa_errors = [
            float(row["pa_mpjpe_mm"])
            for row in frame_records
            if row["camera"] == cam_name and row["pa_mpjpe_mm"] != ""
        ]
        cam_mean_mpjpe = float(np.mean(valid_cam_errors)) if valid_cam_errors else None
        cam_mean_pa_mpjpe = float(np.mean(valid_cam_pa_errors)) if valid_cam_pa_errors else None
        pbar.set_postfix(
            seq=seq_name,
            file=img_name,
            cam=cam_name,
            boxes=num_bbox,
            meshes=len(frame_meshes),
            model_fps=f"{mean_model_fps:.2f}",
            mpjpe=f"{frame_mpjpe:.1f}mm" if frame_mpjpe is not None else "n/a",
            pa=f"{frame_pa_mpjpe:.1f}mm" if frame_pa_mpjpe is not None else "n/a",
            cam_mpjpe=f"{cam_mean_mpjpe:.1f}mm" if cam_mean_mpjpe is not None else "n/a",
            cam_pa=f"{cam_mean_pa_mpjpe:.1f}mm" if cam_mean_pa_mpjpe is not None else "n/a",
            status="saved" if dump_outputs else "processed",
        )

    # print mean inference FPS (detection+model+render per image)
    if all_model_time > 0 and processed_model_frames > 0:
        mean_fps = processed_model_frames / all_model_time
    else:
        mean_fps = 0.0
    print(f"Processed {processed_images} frames. Model-only inference FPS: {mean_fps:.2f}")
    if gt_errors_mm:
        print(f"Mean target-joint MPJPE: {float(np.mean(gt_errors_mm)):.2f} mm over {len(gt_errors_mm)} frames")
    summary_records = summarize_metrics(frame_records)
    final_record = overall_metrics(frame_records)
    summary_records.append(final_record)
    for row in summary_records:
        mpjpe_text = f"{row['mean_mpjpe_mm']} mm" if row["mean_mpjpe_mm"] != "" else "n/a"
        pa_text = f"{row['mean_pa_mpjpe_mm']} mm" if row["mean_pa_mpjpe_mm"] != "" else "n/a"
        mean_fps_text = row["mean_model_fps"] if row["mean_model_fps"] != "" else "n/a"
        overall_fps_text = row["overall_model_fps"] if row["overall_model_fps"] != "" else "n/a"
        print(
            f"{row['sequence']} / {row['camera']}: frames={row['frames']}, "
            f"MPJPE={mpjpe_text}, PA-MPJPE={pa_text}, "
            f"mean_model_fps={mean_fps_text}, overall_model_fps={overall_fps_text}"
        )
    frame_csv, summary_csv = write_metrics_csv(output_path, frame_records, summary_records)
    print(f"Saved metrics CSV: {frame_csv}")
    print(f"Saved summary CSV: {summary_csv}")

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
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    torch.set_float32_matmul_precision('high')
    inference(
        input_path=args.input_path,
        output_path=args.output_path,
        camera_index=args.camera_index,
        end=args.end,
        open3d_vis=args.open3d_vis,
    )
