"""A tiny stand-in for the xgboost package in the browser: evaluates saved tree models with numpy.

firespread.survival only needs Booster.load_model(json file), DMatrix(X, base_margin=...) and
Booster.predict(dmatrix, output_margin=True). The real xgboost for Pyodide is an older version that cannot read
the model files this repo saves, and is a large download; the trees themselves are simple to evaluate.
israel_static/tests/test_xgboost_shim.py checks the numbers against the real xgboost.
"""
from __future__ import annotations

import json

import numpy as np

__version__ = "0-numpy-shim"


class DMatrix:
    def __init__(self, data, label=None, weight=None, base_margin=None, feature_names=None, **_):
        self.data = np.asarray(data, np.float32)
        self.base_margin = None if base_margin is None else np.asarray(base_margin, np.float64)
        self.feature_names = feature_names


class Booster:
    def __init__(self, *_, **__):
        self.trees: list[dict] = []
        self.feature_names: list[str] | None = None

    def load_model(self, fname) -> None:
        with open(fname, encoding="utf-8") as f:
            learner = json.load(f)["learner"]
        booster = learner["gradient_booster"]
        if booster.get("name") != "gbtree":
            raise ValueError(f"only gbtree models are supported, got {booster.get('name')!r}")
        self.feature_names = learner.get("feature_names") or None
        self.trees = []
        for t in booster["model"]["trees"]:
            if any(int(s) != 0 for s in t.get("split_type", [])):
                raise ValueError("categorical splits are not supported")
            left = np.asarray(t["left_children"], np.int64)
            self.trees.append({
                "left": left, "right": np.asarray(t["right_children"], np.int64),
                "feature": np.asarray(t["split_indices"], np.int64),
                "value": np.asarray(t["split_conditions"], np.float32),    # threshold, or the leaf's value
                "default_left": np.asarray(t["default_left"], bool), "leaf": left < 0,
            })

    def predict(self, data: DMatrix, output_margin: bool = False, **_) -> np.ndarray:
        if not output_margin or data.base_margin is None:
            raise NotImplementedError("this stand-in only gives margins on top of a base_margin")
        if self.feature_names and data.feature_names and list(data.feature_names) != list(self.feature_names):
            raise ValueError(f"feature names differ: model {self.feature_names}, data {list(data.feature_names)}")
        X = data.data
        rows = np.arange(X.shape[0])
        out = data.base_margin.astype(np.float64).copy()
        for t in self.trees:
            node = np.zeros(X.shape[0], np.int64)
            while True:
                active = ~t["leaf"][node]
                if not active.any():
                    break
                x = X[rows, t["feature"][node]]
                go_left = np.where(np.isnan(x), t["default_left"][node], x < t["value"][node])
                node = np.where(active, np.where(go_left, t["left"][node], t["right"][node]), node)
            out += t["value"][node]
        return out.astype(np.float32)
