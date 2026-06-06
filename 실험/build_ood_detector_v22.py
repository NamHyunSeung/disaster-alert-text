"""
v22 모델용 OOD 탐지기 구축.

model_v22 + tokenizer_v22 + dedup_v7 train 데이터로
MahalanobisOOD 통계를 계산해 실험/ood_stats_v22.pt 로 저장한다.

실행: 프로젝트 루트에서
  python 실험/build_ood_detector_v22.py
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
from ood_detector import MahalanobisOOD, extract_cls_batch

MODEL_DIR  = os.path.join(ROOT, 'model_v22')
TOK_DIR    = os.path.join(ROOT, 'tokenizer_v22')
DATA_PATH  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                           '재난문자_레이블링결과_dedup_v7.xlsx')
SAVE_PATH  = os.path.join(ROOT, '실험', 'ood_stats_v22.pt')
BATCH_SIZE = 64
MAX_LEN    = 128


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, val_df, _ = load_and_split_v2(DATA_PATH)
    print(f"  학습: {len(train_df)}개, 검증: {len(val_df)}개")

    print("모델 / 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=5)
    model.to(device)
    model.eval()

    train_ds = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("학습 데이터 CLS 임베딩 추출 중...")
    train_embs, train_labels = extract_cls_batch(model, train_loader, device)
    print(f"  임베딩 shape: {train_embs.shape}")

    print("MahalanobisOOD fit 중...")
    detector = MahalanobisOOD()
    detector.fit(train_embs, train_labels)

    min_dists, _ = detector._distances_batch(train_embs)
    detector.threshold = float(np.percentile(min_dists.numpy(), 95))
    print(f"  글로벌 fallback threshold = {detector.threshold:.4f}")

    print("클래스별 threshold 계산 중...")
    detector.fit_class_thresholds(train_embs, train_labels, keep_pct=0.95)

    detector.save(SAVE_PATH)
    print(f"\n저장 완료: {SAVE_PATH}")

    # 클래스별 threshold 출력
    print("\n[클래스별 threshold]")
    for c in range(5):
        thr = detector._threshold_for(c)
        print(f"  L{c}: {thr:.2f}")


if __name__ == '__main__':
    main()
