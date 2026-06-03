"""
OOD 탐지기 구축 스크립트.

model_v9n 모델로 학습 데이터의 CLS 임베딩을 추출하고,
MahalanobisOOD 통계를 계산해 실험/ood_stats.pt 로 저장한다.

실행: 프로젝트 루트에서
  python 실험/build_ood_detector.py

소요 시간: GPU 기준 약 3~5분, CPU 기준 약 15~30분
"""

import sys
import os

# 프로젝트 루트 기준 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from ood_detector import MahalanobisOOD, extract_cls_batch

MODEL_DIR   = os.path.join(ROOT, 'model_v9n')
TOK_DIR     = os.path.join(ROOT, 'tokenizer_v9n')
DATA_PATH   = os.path.join(ROOT, '중요파일', 'data', 'raw',
                            '재난문자_레이블링결과_dedup_v2.xlsx')
SAVE_PATH   = os.path.join(ROOT, '실험', 'ood_stats.pt')
BATCH_SIZE  = 64
MAX_LEN     = 128


def percentile_threshold(detector: MahalanobisOOD,
                          val_embs: torch.Tensor,
                          keep_pct: float = 0.95) -> float:
    """
    검증 세트 임베딩을 기준으로 in-distribution의 keep_pct%를 포함하는
    Mahalanobis 거리 임계값을 계산한다.
    """
    dists = []
    for i in range(val_embs.shape[0]):
        d, _ = detector.score(val_embs[i])
        dists.append(d)
    threshold = float(np.percentile(dists, keep_pct * 100))
    print(f"  거리 분포: min={min(dists):.2f}, "
          f"p50={np.percentile(dists, 50):.2f}, "
          f"p95={np.percentile(dists, 95):.2f}, "
          f"p99={np.percentile(dists, 99):.2f}, "
          f"max={max(dists):.2f}")
    return threshold


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, val_df, _ = load_and_split_v2(DATA_PATH)
    print(f"  학습: {len(train_df)}개, 검증: {len(val_df)}개")

    print("모델 / 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=5
    )
    model.to(device)
    model.eval()

    # 학습 데이터 DataLoader (augment=False — 원문만 사용)
    train_ds = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
    val_ds   = DisasterDatasetAug(val_df,   tokenizer, MAX_LEN, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print("학습 데이터 CLS 임베딩 추출 중...")
    train_embs, train_labels = extract_cls_batch(model, train_loader, device)
    print(f"  임베딩 shape: {train_embs.shape}")

    print("검증 데이터 CLS 임베딩 추출 중...")
    val_embs, val_labels = extract_cls_batch(model, val_loader, device)

    print("MahalanobisOOD fit 중...")
    detector = MahalanobisOOD(threshold=0.0)  # threshold는 아래서 결정
    detector.fit(train_embs, train_labels)

    print("임계값(threshold) 계산 중 (검증 세트 기준 95th percentile)...")
    threshold = percentile_threshold(detector, val_embs, keep_pct=0.95)
    detector.threshold = threshold
    print(f"  최종 threshold = {threshold:.4f}")

    detector.save(SAVE_PATH)
    print(f"저장 완료: {SAVE_PATH}")

    # 간단한 검증
    print("\n[검증] OOD 샘플 검출 테스트")
    from test_ood_distribution_shift import TEST_CASES, mask_text
    from ood_detector import extract_cls_embedding
    for case in TEST_CASES:
        emb  = extract_cls_embedding(model, tokenizer, mask_text(case['text']), device)
        dist, nearest = detector.score(emb)
        ood = dist > detector.threshold
        print(f"  {case['name']:<20} dist={dist:6.2f}  OOD={'YES' if ood else 'no ':3}  nearest=L{nearest}")


if __name__ == '__main__':
    main()
