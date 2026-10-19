"""Artifact cache: runs/<clip>/<stage>.{parquet,npy,json}."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


class Store:
    def __init__(self, clip: str | Path, root: str | Path = "runs"):
        self.dir = Path(root) / Path(clip).stem
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, *names: str) -> bool:
        return all(self.path(n).exists() for n in names)

    def put_df(self, name: str, df: pl.DataFrame) -> None:
        df.write_parquet(self.path(f"{name}.parquet"))

    def df(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.path(f"{name}.parquet"))

    def put_np(self, name: str, a: np.ndarray) -> None:
        np.save(self.path(f"{name}.npy"), a)

    def np(self, name: str) -> np.ndarray:
        return np.load(self.path(f"{name}.npy"))

    def put_json(self, name: str, obj) -> None:
        self.path(f"{name}.json").write_text(json.dumps(obj, indent=1))

    def json(self, name: str):
        return json.loads(self.path(f"{name}.json").read_text())
