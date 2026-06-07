"""
kNN OOD 탐지기 구축 스크립트.

model_v9n으로 학습 데이터의 CLS 임베딩을 추출하고,
KNNOOD 통계를 계산해 실험/knn_ood_stats.pt 로 저장한다.

실행: 프로젝트 루트에서
  python 실험/build_knn_ood.py

소요 시간: 임베딩 추출 GPU ~3분 / CPU ~15분
           kNN threshold 계산 CPU ~1분
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from ood_detector import extract_cls_batch
from ood_knn import KNNOOD

MODEL_DIR  = os.path.join(ROOT, 'model_v9n')
TOK_DIR    = os.path.join(ROOT, 'tokenizer_v9n')
DATA_PATH  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                           '재난문자_레이블링결과_dedup_v2.xlsx')
SAVE_PATH  = os.path.join(ROOT, '실험', 'knn_ood_stats.pt')
BATCH_SIZE = 64
MAX_LEN    = 96
K          = 10


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

    train_ds     = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("학습 데이터 CLS 임베딩 추출 중...")
    train_embs, train_labels = extract_cls_batch(model, train_loader, device)
    print(f"  임베딩 shape: {train_embs.shape}")

    print(f"\nKNNOOD fit 중 (k={K})...")
    detector = KNNOOD(k=K)
    detector.fit(train_embs, train_labels)

    print(f"클래스별 k-NN threshold 계산 중 (95th percentile)...")
    detector.fit_thresholds(train_embs, train_labels, keep_pct=0.95, batch_size=128)

    print(f"\n저장 중: {SAVE_PATH}")
    detector.save(SAVE_PATH)
    size_mb = os.path.getsize(SAVE_PATH) / 1024 / 1024
    print(f"저장 완료: {size_mb:.1f} MB")

    # 학습 데이터 OOD 비율 검증 (5% 전후 예상)
    print("\n[검증] 학습 데이터 OOD 비율 계산 중...")
    mean_dists, nearest_classes = detector._knn_batched(
        train_embs, exclude_self=True, batch_size=128
    )
    ood_count = sum(
        1 for i in range(len(train_embs))
        if mean_dists[i].item() > detector._threshold_for(nearest_classes[i].item())
    )
    n = len(train_embs)
    print(f"  OOD 비율: {ood_count}/{n} ({ood_count/n*100:.1f}%)")
    print("  → 95th percentile threshold 기준이므로 약 5% 예상")


if __name__ == '__main__':
    main()
