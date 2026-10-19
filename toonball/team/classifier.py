"""Team assignment: SigLIP shirt embeddings → UMAP → KMeans(2), majority vote per track."""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, SiglipVisionModel

MODEL = "google/siglip-base-patch16-224"


@torch.no_grad()
def embed(crops: np.ndarray, batch: int = 64) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = SiglipVisionModel.from_pretrained(MODEL).to(device).eval()
    out = []
    for i in range(0, len(crops), batch):
        x = proc(images=list(crops[i:i + batch]), return_tensors="pt").to(device)
        out.append(model(**x).pooler_output.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def cluster(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    import umap

    red = umap.UMAP(n_components=3, random_state=seed).fit_transform(emb)
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(red)


def kit_colour(crops: np.ndarray) -> str:
    """Dominant non-grass colour of a set of shirt crops, as #rrggbb (saturated a little)."""
    c = crops[:, 16:48, 16:48].reshape(-1, 1, 3)  # crop centre is mostly shirt
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    px = c.reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 2] < 160) & (hsv[:, 1] > 50)
    px = px[~grass]
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(px)
    rgb = km.cluster_centers_[np.bincount(km.labels_).argmax()].astype(np.uint8)
    h, s, v = cv2.cvtColor(rgb[None, None], cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    rgb = cv2.cvtColor(np.array([[[h, min(255, s * 1.12), min(255, v * 1.05)]]], np.uint8),
                       cv2.COLOR_HSV2RGB)[0, 0]
    return "#" + "".join(f"{int(x):02x}" for x in rgb)


def assign(crops: np.ndarray, crop_index: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Return (track_id, team, team_conf) and the two kit colours."""
    labels = cluster(embed(crops))
    votes = crop_index.with_columns(pl.Series("lab", labels))
    per_track = (
        votes.group_by("track_id")
        .agg(pl.col("lab").mean().alias("p1"), pl.len().alias("n"))
        .with_columns(
            (pl.col("p1") > 0.5).cast(pl.Int8).alias("team"),
            ((pl.col("p1") - 0.5).abs() * 2).cast(pl.Float32).alias("team_conf"),
        )
        .select("track_id", "team", "team_conf")
    )
    colours = [kit_colour(crops[labels == k]) for k in (0, 1)]
    return per_track, colours
