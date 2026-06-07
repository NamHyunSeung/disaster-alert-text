"""
Mahalanobis Distance 기반 OOD(Out-of-Distribution) 탐지기.

학습 데이터의 클래스별 CLS 임베딩 분포를 기억해두고,
추론 시 입력이 그 분포에서 얼마나 벗어났는지를 Mahalanobis 거리로 측정한다.
클래스별 threshold를 사용해 nearest class 기준으로 OOD를 판정한다.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

NUM_CLASSES = 5
SAFE_FLOOR  = 4   # OOD 감지 시 최대 긴급 등급으로 올림 (미지의 재난 = 최고 위험 가정)


class MahalanobisOOD:
    """
    fit()으로 학습 임베딩을 받아 클래스별 평균 + 공유 역공분산을 계산.
    fit_class_thresholds()로 클래스별 OOD 경계를 학습 데이터 분포에서 결정.
    conservative_predict()로 OOD 여부를 반영한 보수적 예측 레이블을 반환.
    """

    def __init__(self, threshold: float = 50.0):
        self.threshold         = threshold                          # 글로벌 fallback
        self.class_thresholds: Optional[List[float]] = None        # 클래스별 threshold
        self.class_means:      Optional[torch.Tensor] = None       # (C, D)
        self.inv_cov:          Optional[torch.Tensor] = None       # (D, D)
        self._fitted = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor) -> "MahalanobisOOD":
        embeddings = embeddings.float()
        D = embeddings.shape[1]
        means = []
        centered_all = []

        for c in range(NUM_CLASSES):
            mask = labels == c
            if mask.sum() == 0:
                means.append(torch.zeros(D))
                continue
            cls_emb = embeddings[mask]
            mu = cls_emb.mean(0)
            means.append(mu)
            centered_all.append(cls_emb - mu)

        self.class_means = torch.stack(means, dim=0)  # (C, D)

        if centered_all:
            all_centered = torch.cat(centered_all, dim=0)
            cov = (all_centered.T @ all_centered) / (all_centered.shape[0] - 1 + 1e-6)
            cov = cov + 1e-4 * torch.eye(D)
            self.inv_cov = torch.linalg.inv(cov)
        else:
            self.inv_cov = torch.eye(D)

        self._fitted = True
        return self

    def fit_class_thresholds(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        keep_pct: float = 0.95,
    ) -> "MahalanobisOOD":
        """
        클래스별로 학습 샘플의 Mahalanobis 거리 분포를 계산하고
        keep_pct 백분위수를 각 클래스의 threshold로 설정한다.
        nearest class가 c인 샘플의 거리가 class_thresholds[c]를 초과하면 OOD.
        """
        assert self._fitted, "fit()을 먼저 호출하세요"
        # 전체 배치 거리를 벡터화하여 계산 (118K 샘플도 빠름)
        min_dists, nearest_classes = self._distances_batch(embeddings)

        class_thresholds = []
        print("  클래스별 threshold (학습 데이터 기준):")
        for c in range(NUM_CLASSES):
            # nearest class가 c인 학습 샘플들의 거리 분포
            mask = nearest_classes == c
            if mask.sum() == 0:
                class_thresholds.append(self.threshold)
                print(f"    L{c}: 샘플 없음 → fallback {self.threshold:.2f}")
                continue
            dists_c = min_dists[mask].numpy()
            thr = float(np.percentile(dists_c, keep_pct * 100))
            class_thresholds.append(thr)
            print(f"    L{c}: n={mask.sum():5d}  p50={np.percentile(dists_c,50):.2f}"
                  f"  p95={np.percentile(dists_c,95):.2f}  p99={np.percentile(dists_c,99):.2f}"
                  f"  → threshold={thr:.2f}")

        self.class_thresholds = class_thresholds
        return self

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def _distances_batch(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(N,D) 배치에 대해 (min_dist, nearest_class)를 벡터화 계산."""
        emb = embeddings.float().cpu()
        inv = self.inv_cov.float().cpu()
        mu  = self.class_means.float().cpu()

        # all_dists: (N, C)
        all_dists = torch.zeros(emb.shape[0], NUM_CLASSES)
        for c in range(NUM_CLASSES):
            diff = emb - mu[c]              # (N, D)
            d2   = ((diff @ inv) * diff).sum(dim=1)  # (N,)
            all_dists[:, c] = d2.clamp(min=0).sqrt()

        min_dists, nearest = all_dists.min(dim=1)
        return min_dists, nearest

    def score(self, embedding: torch.Tensor) -> Tuple[float, int]:
        """단일 샘플. 반환: (최소 Mahalanobis 거리, nearest class)"""
        assert self._fitted, "fit()을 먼저 호출하세요"
        min_dists, nearest = self._distances_batch(embedding.unsqueeze(0))
        return min_dists[0].item(), nearest[0].item()

    def _threshold_for(self, nearest_class: int) -> float:
        if self.class_thresholds is not None:
            return self.class_thresholds[nearest_class]
        return self.threshold

    def is_ood(self, embedding: torch.Tensor) -> bool:
        dist, nearest = self.score(embedding)
        return dist > self._threshold_for(nearest)

    def conservative_predict(
        self,
        embedding: torch.Tensor,
        base_pred: int,
    ) -> Tuple[int, bool, float]:
        """반환: (최종 예측 레이블, OOD 여부, Mahalanobis 거리)"""
        dist, nearest = self.score(embedding)
        ood = dist > self._threshold_for(nearest)
        final = SAFE_FLOOR if (ood and base_pred < SAFE_FLOOR) else base_pred
        return final, ood, dist

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            'class_means':      self.class_means,
            'inv_cov':          self.inv_cov,
            'threshold':        self.threshold,
            'class_thresholds': self.class_thresholds,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisOOD":
        data = torch.load(path, map_location='cpu')
        obj  = cls(threshold=data['threshold'])
        obj.class_means      = data['class_means']
        obj.inv_cov          = data['inv_cov']
        obj.class_thresholds = data.get('class_thresholds')
        obj._fitted          = True
        return obj


# ------------------------------------------------------------------
# 임베딩 추출 유틸
# ------------------------------------------------------------------

def extract_cls_embedding(model, tokenizer, text: str, device) -> torch.Tensor:
    """모델 last hidden state의 [CLS] 벡터를 반환. shape: (D,)"""
    enc = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=128, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # KoELECTRA: hidden_states[-1] → (1, seq, D)
    cls_vec = out.hidden_states[-1][:, 0, :].squeeze(0).cpu()  # (D,)
    return cls_vec


def extract_cls_batch(model, dataloader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """DataLoader 전체를 순회해 CLS 벡터와 레이블을 반환."""
    all_emb = []
    all_lbl = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ids   = batch['input_ids'].to(device)
            attn  = batch['attention_mask'].to(device)
            ttids = batch.get('token_type_ids')
            if ttids is not None:
                ttids = ttids.to(device)
            kwargs = dict(input_ids=ids, attention_mask=attn,
                          output_hidden_states=True)
            if ttids is not None:
                kwargs['token_type_ids'] = ttids
            out   = model(**kwargs)
            cls   = out.hidden_states[-1][:, 0, :].cpu()  # (B, D)
            all_emb.append(cls)
            all_lbl.append(batch['label'])
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)
