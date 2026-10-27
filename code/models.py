"""Model builders behind a common sklearn-compatible interface."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_model(name: str, seed: int = 0, **kwargs: Any) -> BaseEstimator:
    """Return a fitted-ready sklearn-compatible estimator.

    Parameters
    ----------
    name : str
        One of ``lr``, ``rf``, ``xgboost``, ``ft_transformer``.
    seed : int
    **kwargs
        Passed to the underlying constructor.

    Returns
    -------
    sklearn-compatible estimator with fit / predict / predict_proba.
    """
    name = name.lower()
    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
            **kwargs,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=1,
            **kwargs,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
            **kwargs,
        )
    if name == "ft_transformer":
        return FTTransformerClassifier(seed=seed, **kwargs)
    raise ValueError(f"Unknown model name: {name!r}. Choose lr, rf, xgboost, ft_transformer.")


# ---------------------------------------------------------------------------
# FT-Transformer wrapper
# ---------------------------------------------------------------------------

class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """sklearn wrapper around rtdl FT-Transformer for binary classification.

    Parameters
    ----------
    seed : int
    d_token : int
        Embedding / token dimension.
    n_blocks : int
        Number of transformer blocks.
    attention_dropout : float
    ffn_dropout : float
    residual_dropout : float
    lr : float
        Adam learning rate.
    weight_decay : float
    n_epochs : int
    batch_size : int
    """

    def __init__(
        self,
        seed: int = 0,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 256,
    ) -> None:
        self.seed = seed
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        self._set_seeds()
        import torch
        from rtdl_revisiting_models import FTTransformer

        X_t = self._to_tensor(X)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32))

        n_features = X_t.shape[1]
        default_kwargs = FTTransformer.get_default_kwargs(n_blocks=self.n_blocks)
        self.model_ = FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            d_block=self.d_token,
            attention_n_heads=default_kwargs["attention_n_heads"],
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=default_kwargs["ffn_d_hidden"],
            ffn_d_hidden_multiplier=default_kwargs["ffn_d_hidden_multiplier"],
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            n_blocks=self.n_blocks,
            _is_default=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)
        X_t = X_t.to(device)
        y_t = y_t.to(device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_reduction = "none" if sample_weight is not None else "mean"
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight(y, sample_weight, device),
            reduction=loss_reduction,
        )

        n = len(y_t)
        self.model_.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = X_t[idx]
                y_batch = y_t[idx]
                optimizer.zero_grad()
                logits = self.model_(x_batch, None).squeeze(1)
                if sample_weight is not None:
                    w = torch.tensor(
                        sample_weight[idx.cpu().numpy()], dtype=torch.float32, device=device
                    )
                    batch_loss = (loss_fn(logits, y_batch) * w).mean()
                else:
                    batch_loss = loss_fn(logits, y_batch)
                batch_loss.backward()
                optimizer.step()

        self.device_ = device
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        self.model_.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X).to(self.device_)
            logits = self.model_(X_t, None).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def _to_tensor(X) -> "torch.Tensor":
        import torch

        arr = np.asarray(X, dtype=np.float32)
        return torch.tensor(arr)

    @staticmethod
    def _pos_weight(
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        device: "torch.device",
    ) -> "Optional[torch.Tensor]":
        import torch

        y_arr = np.asarray(y)
        if sample_weight is not None:
            pos_w = sample_weight[y_arr == 0].sum() / (sample_weight[y_arr == 1].sum() + 1e-9)
        else:
            n_neg = (y_arr == 0).sum()
            n_pos = (y_arr == 1).sum()
            pos_w = n_neg / (n_pos + 1e-9)
        return torch.tensor([pos_w], dtype=torch.float32, device=device)
