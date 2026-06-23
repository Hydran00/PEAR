import pandas as pd

path = "outputs/visuotactile_summary_metrics.csv"
df = pd.read_csv(path)
df = df[df["camera"] != "ALL"]

cols = [
    "mean_mpjpe_mm",
    "mean_pa_mpjpe_mm",
    "mean_model_fps",
    "overall_model_fps",
]

camera_means = (
    df.groupby("camera")[cols]
      .mean()
      .round(2)
)

print(camera_means)