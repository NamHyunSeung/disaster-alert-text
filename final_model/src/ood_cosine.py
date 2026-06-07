"""
Cosine Distance 기반 OOD 탐지기.

클래스별 L2 정규화 centroid를 기억하고,
추론 시 입력 CLS 임베딩과의 cosine 거리로 OOD 여부를 판정한다.
클래스별 threshold는 학습 데이터 95th percentile로 결정한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

NUM_CLASSES = 5


class CosineOOD:
    """
    fit()으로 학습 임베딩을 받아 클래스별 L2 정규화 centroid를 계산.
    fit_thresholds()로 클래스별 cosine 거리 95th percentile을 threshold로 설정.
    score()로 (nearest cosine 거리, nearest class)를 반환.
    is_ood()로 OOD 여부를 판정.
    """

    def __init__(self):
        self.class_centroids:  Optional[torch.Tensor] = None  # (C, D) L2 정규화
        self.class_thresholds: Optional[List[float]]  = None  # per-class threshold
        self.global_threshold: float = 0.5                     # fallback
        self._fitted = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor) -> "CosineOOD":
        embeddings = embeddings.float()
        D = embeddings.shape[1]
        centroids = []

        for c in range(NUM_CLASSES):
            mask = labels == c
            if mask.sum() == 0:
                centroids.append(torch.zeros(D))
                continue
            mu = embeddings[mask].mean(0)
            centroids.append(F.normalize(mu, dim=0))  # L2 정규화

        self.class_centroids = torch.stack(centroids, dim=0)  # (C, D)
        self._fitted = True
        return self

    def fit_thresholds(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        keep_pct: float = 0.95,
    ) -> "CosineOOD":
        """클래스별 학습 샘플의 cosine 거리 분포에서 keep_pct 백분위수를 threshold로 설정."""
        assert self._fitted, "fit()을 먼저 호출하세요"
        min_dists, nearest_classes = self._distances_batch(embeddings)

        class_thresholds = []
        print("  클래스별 cosine threshold (학습 데이터 기준):")
        for c in range(NUM_CLASSES):
            mask = nearest_classes == c
            if mask.sum() == 0:
                class_thresholds.append(self.global_threshold)
                print(f"    L{c}: 샘플 없음 → fallback {self.global_threshold:.4f}")
                continue
            dists_c = min_dists[mask].numpy()
            thr = float(np.percentile(dists_c, keep_pct * 100))
            class_thresholds.append(thr)
            print(f"    L{c}: n={mask.sum():5d}  "
                  f"p50={np.percentile(dists_c, 50):.4f}  "
                  f"p95={np.percentile(dists_c, 95):.4f}  "
                  f"p99={np.percentile(dists_c, 99):.4f}  "
                  f"→ threshold={thr:.4f}")

        self.class_thresholds = class_thresholds
        self.global_threshold = float(np.percentile(min_dists.numpy(), keep_pct * 100))
        return self

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def _distances_batch(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(N, D) 배치에 대해 (min_cosine_dist, nearest_class)를 벡터화 계산."""
        emb = F.normalize(embeddings.float().cpu(), dim=1)  # (N, D)
        cen = self.class_centroids.float().cpu()             # (C, D) — 이미 정규화됨

        sim  = emb @ cen.T                  # (N, C) cosine similarity
        dist = (1.0 - sim).clamp(0.0, 2.0)  # (N, C) cosine distance

        min_dists, nearest = dist.min(dim=1)
        return min_dists, nearest

    def score(self, embedding: torch.Tensor) -> Tuple[float, int]:
        """단일 샘플. 반환: (nearest cosine 거리, nearest class)"""
        assert self._fitted, "fit()을 먼저 호출하세요"
        min_dists, nearest = self._distances_batch(embedding.unsqueeze(0))
        return min_dists[0].item(), nearest[0].item()

    def _threshold_for(self, nearest_class: int) -> float:
        if self.class_thresholds is not None:
            return self.class_thresholds[nearest_class]
        return self.global_threshold

    def is_ood(self, embedding: torch.Tensor) -> bool:
        """OOD 여부 반환. 거리·클래스도 필요하면 score()를 별도 호출."""
        dist, nearest = self.score(embedding)
        return dist > self._threshold_for(nearest)

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            'class_centroids':  self.class_centroids,
            'class_thresholds': self.class_thresholds,
            'global_threshold': self.global_threshold,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "CosineOOD":
        data = torch.load(path, map_location='cpu', weights_only=True)
        obj = cls()
        obj.class_centroids  = data['class_centroids']
        obj.class_thresholds = data.get('class_thresholds')
        obj.global_threshold = data.get('global_threshold', 0.5)
        obj._fitted = True
        return obj
