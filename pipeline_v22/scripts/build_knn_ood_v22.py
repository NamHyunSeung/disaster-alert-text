"""
v22 모델용 KNN-OOD 탐지기 구축.

train 임베딩 추출 → 클래스별 KNN 거리 95th percentile threshold 계산
→ 실험/knn_ood_v22.npz + 실험/knn_ood_v22_meta.pt 저장

실행: python 실험/build_knn_ood_v22.py
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from ood_detector import extract_cls_batch

MODEL_DIR  = os.path.join(ROOT, 'model_v22')
TOK_DIR    = os.path.join(ROOT, 'tokenizer_v22')
DATA_PATH  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                           '재난문자_레이블링결과_dedup_v7.xlsx')
EMB_SAVE   = os.path.join(ROOT, '실험', 'knn_ood_v22.npz')
META_SAVE  = os.path.join(ROOT, '실험', 'knn_ood_v22_meta.pt')
BATCH_SIZE = 64
MAX_LEN    = 128
K          = 20   # KNN 이웃 수


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, val_df, _ = load_and_split_v2(DATA_PATH)
    print(f"  학습: {len(train_df)}개")

    print("모델 / 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=5)
    model.to(device).eval()

    train_ds = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("CLS 임베딩 추출 중...")
    train_embs, train_labels = extract_cls_batch(model, train_loader, device)
    embs_np   = train_embs.numpy().astype(np.float32)    # (N, 768)
    labels_np = train_labels.numpy().astype(np.int32)    # (N,)
    print(f"  shape: {embs_np.shape}")

    print(f"KNN fit 중 (K={K}, metric=cosine)...")
    nn = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn.fit(embs_np)

    print("train 샘플별 KNN 거리 계산 중...")
    dists, _ = nn.kneighbors(embs_np)   # (N, K)
    knn_scores = dists.mean(axis=1)      # (N,)

    # 글로벌 threshold
    global_thr = float(np.percentile(knn_scores, 95))
    print(f"  글로벌 threshold (p95) = {global_thr:.6f}")

    # 클래스별 threshold
    class_thresholds = {}
    print("\n[클래스별 threshold]")
    for c in range(5):
        mask = labels_np == c
        scores_c = knn_scores[mask]
        thr_c = float(np.percentile(scores_c, 95))
        class_thresholds[c] = thr_c
        print(f"  L{c}: n={mask.sum():>5}  p50={np.percentile(scores_c,50):.6f}"
              f"  p95={thr_c:.6f}  p99={np.percentile(scores_c,99):.6f}")

    print(f"\n임베딩 저장 중: {EMB_SAVE}")
    np.savez_compressed(EMB_SAVE, embs=embs_np, labels=labels_np)

    print(f"메타데이터 저장 중: {META_SAVE}")
    torch.save({
        'k': K,
        'global_threshold': global_thr,
        'class_thresholds': class_thresholds,
    }, META_SAVE)

    print("\n완료.")


if __name__ == '__main__':
    main()
