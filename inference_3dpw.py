from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


J19_PELVIS_INDEX = 14
J19_3DPW_OC_EVAL_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18],
    dtype=np.int64,
)


def decode_imgname(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def resolve_img_dir(dataset_path: Path, img_dir: str | None) -> Path:
    if img_dir is not None:
        return Path(img_dir)
    return dataset_path.parent


def resolve_image_path(imgname: str, img_dir: Path) -> Path:
    path = Path(imgname)
    return path if path.is_absolute() else img_dir / path


def sequence_name_from_imgname(imgname: str) -> str:
    marker = "imageFiles/"
    idx = imgname.find(marker)
    if idx != -1:
        rest = imgname[idx + len(marker):]
        return rest.split("/", 1)[0]
    return Path(imgname).parent.name


def build_eval_order(
    imgnames: np.ndarray,
    sequence_filter: str | None,
    person_index: int | None,
) -> np.ndarray:
    decoded = [decode_imgname(x) for x in imgnames]
    order = np.arange(len(decoded), dtype=int)

    if sequence_filter is not None:
        order = np.array(
            [i for i, name in enumerate(decoded) if sequence_name_from_imgname(name) == sequence_filter],
            dtype=int,
        )
        if len(order) == 0:
            print(f"[inference_3dpw] sequence_filter='{sequence_filter}' matched no rows; using full dataset")
            order = np.arange(len(decoded), dtype=int)
        else:
            print(f"[inference_3dpw] sequence_filter='{sequence_filter}' selected {len(order)} rows")

    if person_index is not None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for i in order:
            grouped[decoded[int(i)]].append(int(i))
        selected = [
            rows[person_index]
            for _, rows in sorted(grouped.items())
            if 0 <= person_index < len(rows)
        ]
        if selected:
            order = np.asarray(selected, dtype=int)
            print(f"[inference_3dpw] person_index={person_index} selected {len(order)} rows")
        else:
            print(f"[inference_3dpw] person_index={person_index} matched no rows; keeping previous order")

    return order


def rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array(
        [pt_2d[0] * cs - pt_2d[1] * sn, pt_2d[0] * sn + pt_2d[1] * cs],
        dtype=np.float32,
    )


def affine_from_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    dst_width: int,
    dst_height: int,
    inv: bool = False,
) -> np.ndarray:
    src_center = np.array([center_x, center_y], dtype=np.float32)
    src_down = rotate_2d(np.array([0, height * 0.5], dtype=np.float32), 0.0)
    src_right = rotate_2d(np.array([width * 0.5, 0], dtype=np.float32), 0.0)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_down = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_right = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_down, src_center + src_right]).astype(np.float32)
    dst = np.stack([dst_center, dst_center + dst_down, dst_center + dst_right]).astype(np.float32)
    return cv2.getAffineTransform(dst if inv else src, src if inv else dst)


def crop_transform(center: np.ndarray, scale: float, out_size: int = 256, inv: bool = False) -> np.ndarray:
    size = float(scale)
    return affine_from_bbox(
        center_x=float(center[0]),
        center_y=float(center[1]),
        width=size,
        height=size,
        dst_width=out_size,
        dst_height=out_size,
        inv=inv,
    )


def crop_person_patch(
    image_rgb: np.ndarray,
    center: np.ndarray,
    scale: float,
    out_size: int = 256,
) -> np.ndarray:
    trans = crop_transform(center, scale, out_size=out_size)
    return cv2.warpAffine(
        image_rgb,
        trans,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float32)


def load_gt_keypoints(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "extra_keypoints_3d" not in data.files:
        raise KeyError("Expected 'extra_keypoints_3d' in the 3DPW-OC npz.")
    print("[inference_3dpw] loading GT keypoints 'extra_keypoints_3d' once")
    return np.asarray(data["extra_keypoints_3d"])


def gt_j19_from_array(gt_keypoints: np.ndarray, idx: int) -> np.ndarray | None:
    keypoints = np.asarray(gt_keypoints[idx], dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] != 19 or keypoints.shape[1] < 4:
        return None
    valid = keypoints[:, 3] > 0
    if not np.all(valid):
        return None
    return keypoints[:, :3]


def umeyama_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = source.astype(np.float64)
    dst = target.astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    variance = (src_centered ** 2).sum() / src_centered.shape[0]
    covariance = (dst_centered.T @ src_centered) / src_centered.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    scale = np.trace(np.diag(singular_values)) / variance if variance > 0 else 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return (scale * (src @ rotation.T) + translation).astype(np.float32)


def mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(gt - pred, axis=-1).mean())


def pa_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    return mpjpe(gt, umeyama_alignment(pred, gt))


def align_for_mpjpe(
    gt: np.ndarray,
    pred: np.ndarray,
    mode: str,
    pelvis_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return gt, pred
    if mode == "pelvis":
        if pelvis_index is None:
            raise ValueError("pelvis_index is required when mpjpe_align='pelvis'")
        return gt - gt[pelvis_index:pelvis_index + 1], pred - pred[pelvis_index:pelvis_index + 1]
    if mode == "mean":
        return gt - gt.mean(axis=0, keepdims=True), pred - pred.mean(axis=0, keepdims=True)
    return gt - gt[:1], pred - pred[:1]


def select_eval_joints(gt: np.ndarray, pred: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "all":
        return gt, pred
    return gt[J19_3DPW_OC_EVAL_INDICES], pred[J19_3DPW_OC_EVAL_INDICES]


def normalize_device(device_arg: str) -> torch.device | None:
    if device_arg.isdigit():
        device_arg = f"cuda:{device_arg}"
    if not device_arg.startswith("cuda"):
        print("[inference_3dpw] PEAR currently requires CUDA in this repo; pass --device cuda:0.")
        return None
    if not torch.cuda.is_available():
        print("[inference_3dpw] CUDA is not available, but PEAR has CUDA-only code paths in this repo.")
        return None
    return torch.device(device_arg)


def load_pear_model(config_name: str, checkpoint: str | None, device: torch.device):
    from huggingface_hub import hf_hub_download

    from models.modules.ehm import EHM_v2
    from models.pipeline.ehm_pipeline import Ehm_Pipeline
    from utils.general_utils import ConfigDict, add_extra_cfgs

    meta_cfg = ConfigDict(model_config_path=str(Path("configs") / f"{config_name}.yaml"))
    meta_cfg = add_extra_cfgs(meta_cfg)

    if checkpoint is None:
        checkpoint = hf_hub_download(
            repo_id="BestWJH/PEAR_models",
            filename="ehm_model_stage1.pt",
            repo_type="model",
        )

    model = Ehm_Pipeline(meta_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(state["backbone"], strict=False)
    model.head.load_state_dict(state["head"], strict=False)
    model = model.to(device).eval()

    ehm = EHM_v2("assets/FLAME", "assets/SMPLX")
    ehm = ehm.to(device).eval()
    return model, ehm


def load_smplx_to_j19(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    import pickle

    import joblib

    smplx2smpl = joblib.load("assets/SMPLX2SMPL/body_models/smplx2smpl.pkl")["matrix"]
    j_regressor = pickle.load(open("assets/SMPLX2SMPL/SMPL_to_J19.pkl", "rb"), encoding="latin1")
    return (
        torch.as_tensor(smplx2smpl, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(j_regressor, dtype=torch.float32, device=device),
    )


def smplx_vertices_to_j19(
    vertices: torch.Tensor,
    smplx2smpl: torch.Tensor,
    j_regressor: torch.Tensor,
) -> torch.Tensor:
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    vertices = vertices[:, : smplx2smpl.shape[-1]]
    smpl_vertices = torch.matmul(smplx2smpl.expand(vertices.shape[0], -1, -1), vertices)
    return torch.einsum("ji,bik->bjk", j_regressor, smpl_vertices)


def run_pear_on_patch(
    image_patch: np.ndarray,
    model,
    ehm,
    smplx2smpl: torch.Tensor,
    j_regressor: torch.Tensor,
    transform,
    device: torch.device,
    camera_transform: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, torch.Tensor]]:
    image_tensor = transform(image_patch) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        smplx_output = ehm(outputs["body_param"], outputs["flame_param"], pose_type="aa")
        joints = smplx_vertices_to_j19(smplx_output["vertices"], smplx2smpl, j_regressor)
        if camera_transform != "none":
            camera_rt = outputs["pd_cam"]
            if camera_transform == "translation":
                joints = joints + camera_rt[:, None, :3, 3]
            elif camera_transform == "rt":
                joints = torch.matmul(joints, camera_rt[:, :3, :3].transpose(1, 2)) + camera_rt[:, None, :3, 3]
    return joints[0].detach().cpu().numpy().astype(np.float32), outputs, smplx_output


def build_cameras_kwargs(batch_size: int, focal_length: float, device: torch.device) -> dict[str, torch.Tensor | torch.device]:
    screen_size = torch.tensor([1024, 1024], dtype=torch.float32, device=device)[None].repeat(batch_size, 1)
    return {
        "principal_point": torch.zeros(batch_size, 2, dtype=torch.float32, device=device),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": device,
    }


def render_overlay_on_image(
    image_bgr: np.ndarray,
    center: np.ndarray,
    scale: float,
    outputs: dict[str, Any],
    smplx_output: dict[str, torch.Tensor],
    body_renderer,
    lights,
    device: torch.device,
    overlay_alpha: float,
) -> np.ndarray:
    from utils.graphics_utils import GS_Camera

    pd_camera = GS_Camera(
        **build_cameras_kwargs(1, 24, device),
        R=outputs["pd_cam"][:1, :3, :3],
        T=outputs["pd_cam"][:1, :3, 3],
    )
    with torch.inference_mode():
        mesh_rgba = body_renderer.render_mesh(smplx_output["vertices"][None, 0, ...], pd_camera, lights=lights)

    mesh_rgba_np = mesh_rgba.detach().cpu().numpy().clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
    mesh_bgr = cv2.cvtColor(mesh_rgba_np[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
    if mesh_rgba_np.shape[-1] > 3:
        mesh_alpha = mesh_rgba_np[:, :, 3]
    else:
        mesh_alpha = np.any(mesh_bgr > 0, axis=-1).astype(np.uint8) * 255

    mesh_bgr = cv2.resize(mesh_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    mesh_alpha = cv2.resize(mesh_alpha, (256, 256), interpolation=cv2.INTER_AREA)
    inv_trans = crop_transform(center, scale, out_size=256, inv=True)
    height, width = image_bgr.shape[:2]
    mesh_on_orig = cv2.warpAffine(
        mesh_bgr,
        inv_trans,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha_on_orig = cv2.warpAffine(
        mesh_alpha,
        inv_trans,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    vis = image_bgr.copy()
    mask = alpha_on_orig > 10
    vis[mask] = (
        overlay_alpha * mesh_on_orig[mask].astype(np.float32)
        + (1.0 - overlay_alpha) * vis[mask].astype(np.float32)
    ).astype(np.uint8)
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEAR on 3DPW-OC and report MPJPE, PA-MPJPE, and FPS.")
    parser.add_argument("--dataset", type=str, default="data/datasets/3dpw_test_oc.npz")
    parser.add_argument("--img-dir", type=str, default=None, help="Directory containing imageFiles/.")
    parser.add_argument("--config-name", "-c", type=str, default="infer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local PEAR checkpoint.")
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--sequence-filter", type=str, default=None)
    parser.add_argument("--person-index", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--mpjpe-align", choices=("pelvis", "root0", "mean", "none"), default="pelvis")
    parser.add_argument("--pelvis-index", type=int, default=J19_PELVIS_INDEX)
    parser.add_argument("--eval-joints", choices=("3dpw_oc", "all"), default="3dpw_oc")
    parser.add_argument("--camera-transform", choices=("none", "translation", "rt"), default="none")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Show realtime SMPL-X overlay on the original 3DPW image.")
    parser.add_argument("--vis-output-dir", type=str, default=None, help="Optional directory where overlay frames are saved.")
    parser.add_argument("--overlay-alpha", type=float, default=0.65)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--wait-key", type=int, default=1, help="cv2.waitKey delay in ms for realtime visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    device = normalize_device(args.device)
    if device is None:
        return
    if args.mpjpe_align == "pelvis" and args.pelvis_index is None:
        print("[inference_3dpw] --mpjpe-align pelvis requires --pelvis-index.")
        return

    data = np.load(str(dataset_path), allow_pickle=True)
    try:
        imgnames = np.asarray(data["imgname"])
        centers = np.asarray(data["center"], dtype=np.float32)
        scales = np.asarray(data["scale"], dtype=np.float32)
        gt_keypoints = load_gt_keypoints(data)

        print(f"[inference_3dpw] loaded dataset={dataset_path} rows={len(imgnames)}")
        order = build_eval_order(imgnames, args.sequence_filter, args.person_index)
        start = max(0, args.start)
        end = len(order) if args.end <= 0 or args.end > len(order) else args.end
        order = order[start:end]
        if len(order) == 0:
            print(f"[inference_3dpw] empty evaluation range start={start} end={end}")
            return

        img_dir = resolve_img_dir(dataset_path, args.img_dir)
        first_image = resolve_image_path(decode_imgname(imgnames[int(order[0])]), img_dir)
        if not first_image.exists():
            print(f"[inference_3dpw] first image not found: {first_image}")
            print("[inference_3dpw] pass --img-dir pointing to the directory that contains imageFiles/.")
            return

        print(f"[inference_3dpw] using img_dir={img_dir}")
        print(f"[inference_3dpw] loading PEAR on {device}")
        torch.set_float32_matmul_precision("high")
        model, ehm = load_pear_model(args.config_name, args.checkpoint, device)
        smplx2smpl, j_regressor = load_smplx_to_j19(device)
        transform = transforms.ToTensor()
        body_renderer = None
        lights = None
        vis_output_dir = Path(args.vis_output_dir) if args.vis_output_dir is not None else None
        if args.visualize or vis_output_dir is not None:
            from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
            from pytorch3d.renderer import PointLights

            body_renderer = BodyRenderer("assets/SMPLX", args.render_size, focal_length=24.0).to(device)
            body_renderer.eval()
            lights = PointLights(device=device, location=[[0.0, -1.0, -10.0]])
            if vis_output_dir is not None:
                vis_output_dir.mkdir(parents=True, exist_ok=True)

        mpjpes: list[float] = []
        pa_mpjpes: list[float] = []
        skipped = 0
        processed = 0
        total_time = 0.0
        model_time = 0.0

        for local_pos, dataset_idx in enumerate(order):
            row = int(dataset_idx)
            frame_pos = start + local_pos
            gt = gt_j19_from_array(gt_keypoints, row)
            if gt is None:
                skipped += 1
                print(f"Frame {frame_pos} (row {row}): invalid GT J19, skipping")
                continue

            img_path = resolve_image_path(decode_imgname(imgnames[row]), img_dir)
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image_bgr is None:
                skipped += 1
                print(f"Frame {frame_pos} (row {row}): could not read image {img_path}, skipping")
                continue

            t0 = time.perf_counter()
            image_rgb = image_bgr[:, :, ::-1].astype(np.float32)
            image_patch = crop_person_patch(image_rgb, centers[row], float(scales[row]))

            torch.cuda.synchronize(device)
            model_start = time.perf_counter()
            pred, outputs, smplx_output = run_pear_on_patch(
                image_patch,
                model,
                ehm,
                smplx2smpl,
                j_regressor,
                transform,
                device,
                args.camera_transform,
            )
            torch.cuda.synchronize(device)
            model_end = time.perf_counter()

            if body_renderer is not None and lights is not None:
                vis_img = render_overlay_on_image(
                    image_bgr=image_bgr,
                    center=centers[row],
                    scale=float(scales[row]),
                    outputs=outputs,
                    smplx_output=smplx_output,
                    body_renderer=body_renderer,
                    lights=lights,
                    device=device,
                    overlay_alpha=args.overlay_alpha,
                )
                cv2.putText(
                    vis_img,
                    f"row {row}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if args.visualize:
                    cv2.imshow("PEAR 3DPW-OC overlay", vis_img)
                    key = cv2.waitKey(args.wait_key) & 0xFF
                    if key in (ord("q"), 27):
                        print("[inference_3dpw] visualization stopped by user")
                        break
                if vis_output_dir is not None:
                    cv2.imwrite(str(vis_output_dir / f"overlay_{frame_pos:06d}_row_{row:06d}.jpg"), vis_img)

            total_time += time.perf_counter() - t0
            model_time += model_end - model_start
            processed += 1

            gt_aligned, pred_aligned = align_for_mpjpe(gt, pred, args.mpjpe_align, args.pelvis_index)
            gt_eval, pred_eval = select_eval_joints(gt_aligned, pred_aligned, args.eval_joints)
            gt_pa, pred_pa = select_eval_joints(gt, pred, args.eval_joints)
            frame_mpjpe = mpjpe(gt_eval, pred_eval)
            frame_pa_mpjpe = pa_mpjpe(gt_pa, pred_pa)
            mpjpes.append(frame_mpjpe)
            pa_mpjpes.append(frame_pa_mpjpe)

            if args.print_every > 0 and (processed == 1 or processed % args.print_every == 0):
                fps = processed / total_time if total_time > 0 else 0.0
                model_fps = processed / model_time if model_time > 0 else 0.0
                print(
                    f"Frame {frame_pos} (row {row}): "
                    f"MPJPE={frame_mpjpe * 1000.0:.1f}mm, "
                    f"PA-MPJPE={frame_pa_mpjpe * 1000.0:.1f}mm, "
                    f"FPS={fps:.2f}, model_FPS={model_fps:.2f}"
                )
                if args.diagnostics:
                    gt_root0, pred_root0 = select_eval_joints(
                        *align_for_mpjpe(gt, pred, "root0", None),
                        args.eval_joints,
                    )
                    gt_pelvis, pred_pelvis = select_eval_joints(
                        *align_for_mpjpe(gt, pred, "pelvis", args.pelvis_index),
                        args.eval_joints,
                    )
                    gt_mean, pred_mean = select_eval_joints(
                        *align_for_mpjpe(gt, pred, "mean", None),
                        args.eval_joints,
                    )
                    gt_none, pred_none = select_eval_joints(gt, pred, args.eval_joints)
                    print(
                        f"  diagnostics: root0={mpjpe(gt_root0, pred_root0) * 1000.0:.1f}mm, "
                        f"pelvis={mpjpe(gt_pelvis, pred_pelvis) * 1000.0:.1f}mm, "
                        f"mean={mpjpe(gt_mean, pred_mean) * 1000.0:.1f}mm, "
                        f"none={mpjpe(gt_none, pred_none) * 1000.0:.1f}mm"
                    )

        if not mpjpes:
            print(f"No valid predictions found in range [{start},{end}) - skipped {skipped} frames")
            return

        fps = processed / total_time if total_time > 0 else 0.0
        model_fps = processed / model_time if model_time > 0 else 0.0
        print(f"Processed frames: {processed}, skipped: {skipped}")
        print(
            f"Overall MPJPE ({args.mpjpe_align}-aligned, eval_joints={args.eval_joints}, "
            f"camera_transform={args.camera_transform}): "
            f"{float(np.mean(mpjpes)) * 1000.0:.2f} mm"
        )
        print(f"Overall PA-MPJPE: {float(np.mean(pa_mpjpes)) * 1000.0:.2f} mm")
        print(f"Overall FPS: {fps:.2f}")
        print(f"Model-only FPS: {model_fps:.2f}")
    finally:
        if "args" in locals() and getattr(args, "visualize", False):
            cv2.destroyAllWindows()
        data.close()


if __name__ == "__main__":
    main()
