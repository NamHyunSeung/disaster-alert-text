"""
Mahalanobis Distance 기반 OOD(Out-of-Distribution) 탐지기.

학습 데이터의 클래스별 CLS 임베딩 분포를 기억해두고,
추론 시 입력이 그 분포에서 얼마나 벗어났는지를 Mahalanobis 거리로 측정한다.
거리가 threshold를 초과하면 OOD로 간주하고 안전을 위해 최소 L3으로 상향한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

NUM_CLASSES = 5
SAFE_FLOOR   = 4   # OOD 감지 시 최대 긴급 등급으로 올림 (미지의 재난 = 최고 위험 가정)


class MahalanobisOOD:
    """
    fit()으로 학습 임베딩을 받아 클래스별 평균 + 공유 역공분산을 계산.
    score()로 새 임베딩의 Mahalanobis 거리를 반환.
    conservative_predict()로 OOD 여부를 반영한 보수적 예측 레이블을 반환.
    """

    def __init__(self, threshold: float = 50.0):
        self.threshold   = threshold
        self.class_means: Optional[torch.Tensor] = None   # (C, D)
        self.inv_cov:     Optional[torch.Tensor] = None   # (D, D)
        self._fitted = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor) -> "MahalanobisOOD":
        """
        embeddings: (N, D) float32 — CLS 벡터
        labels:     (N,)   long    — 0~4 클래스
        """
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
            all_centered = torch.cat(centered_all, dim=0)  # (N, D)
            cov = (all_centered.T @ all_centered) / (all_centered.shape[0] - 1 + 1e-6)
            # 수치 안정성: ridge regularization
            cov = cov + 1e-4 * torch.eye(D)
            self.inv_cov = torch.linalg.inv(cov)  # (D, D)
        else:
            self.inv_cov = torch.eye(D)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def score(self, embedding: torch.Tensor) -> Tuple[float, int]:
        """
        embedding: (D,) — 단일 샘플의 CLS 벡터
        반환: (최소 Mahalanobis 거리, 가장 가까운 클래스)
        """
        assert self._fitted, "fit()을 먼저 호출하세요"
        embedding = embedding.float().cpu()
        inv_cov   = self.inv_cov.float().cpu()
        means     = self.class_means.float().cpu()

        min_dist = float('inf')
        nearest  = 0
        for c in range(NUM_CLASSES):
            diff = embedding - means[c]  # (D,)
            d2   = (diff @ inv_cov @ diff).item()
            dist = d2 ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest  = c
        return min_dist, nearest

    def is_ood(self, embedding: torch.Tensor) -> bool:
        dist, _ = self.score(embedding)
        return dist > self.threshold

    def conservative_predict(
        self,
        embedding: torch.Tensor,
        base_pred: int,
    ) -> Tuple[int, bool, float]:
        """
        OOD 탐지 후 보수적 예측 반환.
        반환: (최종 예측 레이블, OOD 여부, Mahalanobis 거리)
        """
        dist, _ = self.score(embedding)
        ood = dist > self.threshold
        if ood and base_pred < SAFE_FLOOR:
            final = SAFE_FLOOR
        else:
            final = base_pred
        return final, ood, dist

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            'class_means': self.class_means,
            'inv_cov':     self.inv_cov,
            'threshold':   self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisOOD":
        data = torch.load(path, map_location='cpu')
        obj  = cls(threshold=data['threshold'])
        obj.class_means = data['class_means']
        obj.inv_cov     = data['inv_cov']
        obj._fitted     = True
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
