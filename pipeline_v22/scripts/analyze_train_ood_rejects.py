"""
1,2단계: 새 threshold(LOO + val 보정) 기준으로 '학습 데이터인데 OOD로 거부되는 샘플'을
직접 들여다봐서 - 모델이 실제로도 헷갈려하는(오분류) 데이터인지, 멀쩡한 정상 문장인데
억울하게 거부되는 건지 - 분류 방향(완화 vs 데이터 정제)을 판단하기 위한 분석.

실행: python pipeline_v22/scripts/analyze_train_ood_rejects.py
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pipeline_v22
PROJ = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(PROJ, '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

from dataset_v2 import load_and_split_v2, DisasterDatasetAug

MODEL_DIR  = os.path.join(ROOT, 'model')
TOK_DIR    = os.path.join(ROOT, 'tokenizer')
DATA_PATH  = os.path.join(PROJ, '중요파일', 'data', 'raw',
                          '재난문자_레이블링결과_dedup_v7.xlsx')
NEW_META   = os.path.join(ROOT, 'ood', 'knn_ood_v22_meta_recalibrated.pt')
OUT_TXT    = os.path.join(ROOT, 'results', 'train_ood_reject_analysis.txt')
BATCH_SIZE = 64
MAX_LEN    = 128
K          = 20
LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']


def extract(model, loader, device):
    embs, labels, preds, confs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            ttids = batch.get('token_type_ids')
            kwargs = dict(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            if ttids is not None:
                kwargs['token_type_ids'] = ttids.to(device)
            out = model(**kwargs)
            probs = F.softmax(out.logits, dim=-1)
            embs.append(out.hidden_states[-1][:, 0, :].cpu())
            labels.append(batch['label'])
            preds.append(probs.argmax(dim=-1).cpu())
            confs.append(probs.max(dim=-1).values.cpu())
    return (torch.cat(embs).numpy().astype(np.float32),
            torch.cat(labels).numpy().astype(np.int32),
            torch.cat(preds).numpy().astype(np.int32),
            torch.cat(confs).numpy().astype(np.float32))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, _, _ = load_and_split_v2(DATA_PATH)
    texts = train_df['text'].tolist()
    n = len(train_df)

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    loader = DataLoader(DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("CLS 임베딩 / 예측 추출 중...")
    embs, labels, preds, confs = extract(model, loader, device)
    print(f"  shape: {embs.shape}")

    print(f"KNN 인덱스 fit 및 LOO score 계산 (K={K})...")
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn.fit(embs)
    dists, _ = nn.kneighbors(embs)
    scores = dists[:, 1:].mean(axis=1)   # 자기 자신 제외 LOO score

    meta = torch.load(NEW_META, weights_only=False)
    class_thr  = meta['class_thresholds']
    global_thr = meta['global_threshold']
    thr_arr = np.array([class_thr.get(int(p), global_thr) for p in preds])
    is_ood = scores > thr_arr

    misclassified = preds != labels

    n_ood = int(is_ood.sum())
    n_ood_mis = int((is_ood & misclassified).sum())
    n_ood_correct = n_ood - n_ood_mis
    n_in_mis = int((~is_ood & misclassified).sum())

    lines = []
    lines.append("=" * 80)
    lines.append("[1단계] OOD로 거부된 학습 샘플 분석")
    lines.append("=" * 80)
    lines.append(f"전체 학습 샘플          : {n}")
    lines.append(f"OOD로 거부된 샘플       : {n_ood}  ({n_ood/n*100:.2f}%)")
    lines.append(f"  - 그 중 모델도 오분류 : {n_ood_mis}  (거부 샘플의 {n_ood_mis/max(n_ood,1)*100:.1f}%)")
    lines.append(f"  - 그 중 모델은 정분류 : {n_ood_correct}  (거부 샘플의 {n_ood_correct/max(n_ood,1)*100:.1f}%)")
    lines.append(f"(참고) 전체 학습 데이터 자체의 오분류율 : {misclassified.sum()}/{n} ({misclassified.mean()*100:.2f}%)")
    lines.append(f"(참고) OOD로 거부 안 된 샘플 중 오분류  : {n_in_mis}  ({n_in_mis/max((~is_ood).sum(),1)*100:.2f}%)")
    lines.append("")
    lines.append(f"=> 전체 오분류율({misclassified.mean()*100:.2f}%) 대비, OOD로 거부된 샘플의 오분류율"
                 f"({n_ood_mis/max(n_ood,1)*100:.1f}%)이 훨씬 높다면 -> OOD필터가 '모델이 헷갈려하는"
                 f" 애매한 데이터'를 잘 짚어내고 있다는 뜻 (방향 B: 데이터 정제 검토)")
    lines.append(f"   비슷하거나 낮다면 -> 멀쩡한 정상 데이터까지 억울하게 거부하는 비중이 크다는 뜻"
                 f" (방향 A: threshold 완화)")
    lines.append("")

    # 클래스별 분포
    lines.append("[클래스별 거부 비율 (정답 레이블 기준)]")
    for c in range(5):
        mask_c = labels == c
        ood_c = is_ood & mask_c
        lines.append(f"  L{c}: 전체 {mask_c.sum():>6}개 중 거부 {ood_c.sum():>5}개 "
                     f"({ood_c.sum()/max(mask_c.sum(),1)*100:.2f}%)  "
                     f"-> 그 중 오분류 {int((ood_c & misclassified).sum())}개")
    lines.append("")

    # 샘플 텍스트 들여다보기
    rng = np.random.default_rng(42)

    def sample_block(title, mask, k=15):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return [f"[{title}] 해당 없음"]
        pick = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
        out = [f"[{title}]  (전체 {len(idxs)}개 중 {len(pick)}개 샘플링)"]
        for i in pick:
            out.append(f"  - 정답=L{labels[i]} 예측=L{preds[i]}({confs[i]*100:4.1f}%) "
                       f"score={scores[i]:.6f} thr={thr_arr[i]:.6f} | {texts[i][:70]}")
        return out

    lines.append("=" * 80)
    lines.append("[샘플 들여다보기]")
    lines.append("=" * 80)
    lines += sample_block("A. OOD로 거부 + 모델도 오분류 (헷갈리는/애매한 데이터일 가능성)",
                          is_ood & misclassified)
    lines.append("")
    lines += sample_block("B. OOD로 거부 + 모델은 정분류 (멀쩡한데 억울하게 거부됐을 가능성)",
                          is_ood & ~misclassified)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)
    print(f"\n결과 저장: {OUT_TXT}")


if __name__ == '__main__':
    main()
