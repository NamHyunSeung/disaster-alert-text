"""
kNN Distance 기반 OOD 탐지기.

클래스 centroid 대신 학습 데이터 전체의 k-최근접 이웃까지의
평균 cosine 거리로 OOD 여부를 판정한다.
centroid 방식으로는 잡지 못하는 '분포 밀도 바깥' 케이스를 포착할 수 있다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

NUM_CLASSES = 5


class KNNOOD:
    """
    fit()으로 학습 임베딩 전체를 저장.
    fit_thresholds()로 클래스별 k-NN 거리 95th percentile을 threshold로 설정.
    score()로 (mean k-NN cosine 거리, k-NN 다수결 클래스)를 반환.
    is_ood()로 OOD 여부를 판정.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.train_embs:       Optional[torch.Tensor] = None  # (N, D) L2 정규화
        self.train_labels:     Optional[torch.Tensor] = None  # (N,)
        self.class_thresholds: Optional[List[float]]  = None
        self.global_threshold: float = 0.5
        self._fitted = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor) -> "KNNOOD":
        self.train_embs   = F.normalize(embeddings.float(), dim=1).cpu()  # (N, D)
        self.train_labels = labels.long().cpu()                            # (N,)
        self._fitted = True
        return self

    def fit_thresholds(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        keep_pct: float = 0.95,
        batch_size: int = 128,
    ) -> "KNNOOD":
        """학습 샘플마다 자기 자신을 제외한 k-NN 거리를 계산하고 클래스별 95th percentile을 threshold로 설정."""
        assert self._fitted, "fit()을 먼저 호출하세요"
        print(f"  학습 데이터 k={self.k}-NN 거리 계산 중 (leave-one-out)...")
        mean_dists, _ = self._knn_batched(embeddings, exclude_self=True, batch_size=batch_size)

        class_thresholds = []
        print("  클래스별 kNN threshold:")
        for c in range(NUM_CLASSES):
            mask = labels == c
            if mask.sum() == 0:
                class_thresholds.append(self.global_threshold)
                print(f"    L{c}: 샘플 없음 → fallback {self.global_threshold:.4f}")
                continue
            d = mean_dists[mask].numpy()
            thr = float(np.percentile(d, keep_pct * 100))
            class_thresholds.append(thr)
            print(f"    L{c}: n={mask.sum():5d}  "
                  f"p50={np.percentile(d, 50):.4f}  "
                  f"p95={np.percentile(d, 95):.4f}  "
                  f"p99={np.percentile(d, 99):.4f}  "
                  f"→ threshold={thr:.4f}")

        self.class_thresholds = class_thresholds
        self.global_threshold = float(np.percentile(mean_dists.numpy(), keep_pct * 100))
        return self

    # ------------------------------------------------------------------
    # 내부 연산
    # ------------------------------------------------------------------

    def _knn_single_batch(
        self,
        queries: torch.Tensor,  # (M, D) — L2 정규화
        exclude_self: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """단일 배치 kNN 계산. 반환: (mean_dist (M,), majority_class (M,))"""
        k_fetch = self.k + 1 if exclude_self else self.k

        sim  = queries @ self.train_embs.T          # (M, N)
        dist = (1.0 - sim).clamp(0.0, 2.0)          # (M, N) cosine distance

        topk_dists, topk_idx = dist.topk(k_fetch, dim=1, largest=False)  # (M, k_fetch)

        if exclude_self:
            # 학습 데이터 자기 자신 제거: 가장 가까운 항목(거리≈0) 제외
            topk_dists = topk_dists[:, 1:]
            topk_idx   = topk_idx[:, 1:]

        mean_dists      = topk_dists.mean(dim=1)               # (M,)
        labels_k        = self.train_labels[topk_idx]          # (M, k)
        nearest_classes = labels_k.mode(dim=1).values          # (M,) — 다수결

        return mean_dists, nearest_classes

    def _knn_batched(
        self,
        embeddings: torch.Tensor,
        exclude_self: bool = False,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = F.normalize(embeddings.float(), dim=1).cpu()
        all_dists, all_classes = [], []
        total = len(queries)
        for start in range(0, total, batch_size):
            q = queries[start : start + batch_size]
            d, c = self._knn_single_batch(q, exclude_self=exclude_self)
            all_dists.append(d)
            all_classes.append(c)
            print(f"\r    {min(start + batch_size, total)}/{total}", end='', flush=True)
        print()
        return torch.cat(all_dists), torch.cat(all_classes)

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def score(self, embedding: torch.Tensor) -> Tuple[float, int]:
        """단일 샘플. 반환: (mean k-NN cosine 거리, kNN 다수결 클래스)"""
        assert self._fitted, "fit()을 먼저 호출하세요"
        q = F.normalize(embedding.float().unsqueeze(0), dim=1).cpu()
        d, c = self._knn_single_batch(q, exclude_self=False)
        return d[0].item(), c[0].item()

    def _threshold_for(self, nearest_class: int) -> float:
        if self.class_thresholds is not None:
            return self.class_thresholds[nearest_class]
        return self.global_threshold

    def is_ood(self, embedding: torch.Tensor) -> bool:
        dist, nearest = self.score(embedding)
        return dist > self._threshold_for(nearest)

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            'train_embs':       self.train_embs,
            'train_labels':     self.train_labels,
            'k':                self.k,
            'class_thresholds': self.class_thresholds,
            'global_threshold': self.global_threshold,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "KNNOOD":
        data = torch.load(path, map_location='cpu', weights_only=True)
        obj = cls(k=data['k'])
        obj.train_embs       = data['train_embs']
        obj.train_labels     = data['train_labels']
        obj.class_thresholds = data.get('class_thresholds')
        obj.global_threshold = data.get('global_threshold', 0.5)
        obj._fitted = True
        return obj
