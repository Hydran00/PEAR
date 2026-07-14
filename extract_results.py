from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def camera_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_cam_1_metrics"):
        return "cam_1"
    if name.endswith("_cam_2_metrics"):
        return "cam_2"
    if "_cam_" in name:
        return "cam_" + name.rsplit("_cam_", 1)[1].replace("_metrics", "")
    return "unknown"


csv_files = sorted(
    path
    for path in OUTPUT_DIR.glob("*_metrics.csv")
    if not path.name.startswith("visuotactile_")
)

if not csv_files:
    raise FileNotFoundError(f"Nessun file *_metrics.csv trovato in {OUTPUT_DIR}")

rows = []
volume_rows = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    if "mpjpe_mm" not in df.columns or "pa_mpjpe_mm" not in df.columns:
        continue

    df["mpjpe_mm"] = pd.to_numeric(df["mpjpe_mm"], errors="coerce")
    df["pa_mpjpe_mm"] = pd.to_numeric(df["pa_mpjpe_mm"], errors="coerce")
    df = df.dropna(subset=["mpjpe_mm", "pa_mpjpe_mm"], how="all")

    if df.empty:
        continue

    df["source_file"] = csv_file.name
    df["camera"] = camera_from_filename(csv_file)
    rows.append(df[["source_file", "camera", "mpjpe_mm", "pa_mpjpe_mm"]])

    volume_cols = [
        "smpl_v2v_mm",
        "smpl_chamfer_mm",
        "abcd_marker_surface_mae_mm",
    ]
    if "frame" in df.columns and all(col in df.columns for col in volume_cols):
        volume_df = df[["frame", "source_file", "camera", *volume_cols]].copy()
        volume_df["frame"] = pd.to_numeric(volume_df["frame"], errors="coerce")
        for col in volume_cols:
            volume_df[col] = pd.to_numeric(volume_df[col], errors="coerce")
        volume_df = volume_df[volume_df["frame"].isin(range(100, 1501, 100))]
        volume_df = volume_df.dropna(subset=volume_cols, how="all")
        if not volume_df.empty:
            volume_rows.append(volume_df)

if not rows:
    raise ValueError("Nessun valore MPJPE / PA-MPJPE valido trovato nei CSV.")

all_metrics = pd.concat(rows, ignore_index=True)

overall_mpjpe = all_metrics["mpjpe_mm"].mean()
overall_pa_mpjpe = all_metrics["pa_mpjpe_mm"].mean()

print("\n================ METRICHE MEDIE ================")
print(f"File letti: {len(csv_files)}")
print(f"Frame validi MPJPE: {all_metrics['mpjpe_mm'].notna().sum()}")
print(f"Frame validi PA-MPJPE: {all_metrics['pa_mpjpe_mm'].notna().sum()}")
print(f"MPJPE medio: {overall_mpjpe:.2f} mm")
print(f"PA-MPJPE medio: {overall_pa_mpjpe:.2f} mm")

print("\n================ MEDIE PER CAMERA ================")
by_camera = (
    all_metrics.groupby("camera")
    .agg(
        frames=("source_file", "count"),
        mean_mpjpe_mm=("mpjpe_mm", "mean"),
        mean_pa_mpjpe_mm=("pa_mpjpe_mm", "mean"),
    )
    .round(2)
)
print(by_camera.to_string())
print("=================================================")

if volume_rows:
    all_volume_metrics = pd.concat(volume_rows, ignore_index=True)
    by_stage = (
        all_volume_metrics.groupby("frame")
        .agg(
            samples=("source_file", "count"),
            mean_v2v_mm=("smpl_v2v_mm", "mean"),
            mean_cd_mm=("smpl_chamfer_mm", "mean"),
            mean_me_mm=("abcd_marker_surface_mae_mm", "mean"),
        )
        .reindex(range(100, 1501, 100))
        .reset_index()
        .rename(columns={"frame": "stage_frame"})
    )
    output_csv = OUTPUT_DIR / "volumetric_stage_metrics.csv"
    by_stage.to_csv(output_csv, index=False)

    print("\n================ VOLUMETRICHE PER STAGE ================")
    print(by_stage.round(2).to_string(index=False))
    print(f"CSV salvato: {output_csv}")
    print("========================================================")
else:
    print("\nNessuna metrica volumetrica valida trovata per frame 100..1500.")
