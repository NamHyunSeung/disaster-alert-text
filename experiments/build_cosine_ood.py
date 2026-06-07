"""
Cosine OOD 탐지기 구축 스크립트.

model_v9n으로 학습 데이터의 CLS 임베딩을 추출하고,
CosineOOD 통계를 계산해 실험/cosine_ood_stats.pt 로 저장한다.

실행: 프로젝트 루트에서
  python 실험/build_cosine_ood.py

소요 시간: GPU 기준 약 3~5분, CPU 기준 약 15~30분
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from ood_detector import extract_cls_batch
from ood_cosine import CosineOOD

MODEL_DIR  = os.path.join(ROOT, 'model_v9n')
TOK_DIR    = os.path.join(ROOT, 'tokenizer_v9n')
DATA_PATH  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                           '재난문자_레이블링결과_dedup_v2.xlsx')
SAVE_PATH  = os.path.join(ROOT, '실험', 'cosine_ood_stats.pt')
BATCH_SIZE = 64
MAX_LEN    = 96


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, val_df, _ = load_and_split_v2(DATA_PATH)
    print(f"  학습: {len(train_df)}개, 검증: {len(val_df)}개")

    print("모델 / 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    train_ds = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("학습 데이터 CLS 임베딩 추출 중...")
    train_embs, train_labels = extract_cls_batch(model, train_loader, device)
    print(f"  임베딩 shape: {train_embs.shape}")

    print("CosineOOD fit 중...")
    detector = CosineOOD()
    detector.fit(train_embs, train_labels)

    print("클래스별 cosine threshold 계산 중 (95th percentile)...")
    detector.fit_thresholds(train_embs, train_labels, keep_pct=0.95)

    detector.save(SAVE_PATH)
    print(f"\n저장 완료: {SAVE_PATH}")

    # 학습 데이터 OOD 비율 확인 (5% 전후 예상)
    min_dists, nearest = detector._distances_batch(train_embs)
    ood_flags = torch.tensor([
        min_dists[i].item() > detector._threshold_for(nearest[i].item())
        for i in range(len(train_embs))
    ])
    n_ood = ood_flags.sum().item()
    print(f"\n[검증] 학습 데이터 OOD 비율: {n_ood}/{len(train_embs)} ({n_ood/len(train_embs)*100:.1f}%)")
    print("  → 95th percentile threshold 기준이므로 약 5% 예상")


if __name__ == '__main__':
    main()
