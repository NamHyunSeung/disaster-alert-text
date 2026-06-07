"""
model_v19 + dedup_v7 학습 데이터 임베딩 기반 kNN 이웃 확인.
신종호흡기/바이러스변종의 최근접 이웃이 L0→L1/L2로 바뀌었는지 검증.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_v2 import load_and_split_v2, DisasterDatasetAug

MODEL_DIR = os.path.join(ROOT, 'model_v19')
TOK_DIR   = os.path.join(ROOT, 'tokenizer')
DATA_PATH = os.path.join(ROOT, '중요파일', 'data', 'raw',
                         '재난문자_레이블링결과_dedup_v7.xlsx')
MAX_LEN   = 128
EMB_BATCH = 256

QUERIES = [
    ("바이러스변종 (정답 L2, v9n→L0, v19→?)",
     "바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고."),
    ("신종호흡기 (정답 L1, v9n→L0, v19→?)",
     "신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화."),
]

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("데이터 / 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

train_df, _, _ = load_and_split_v2(DATA_PATH)
print(f"  학습 세트: {len(train_df)}개")

# 학습 데이터 임베딩 추출
train_ds = DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False)
train_loader = DataLoader(train_ds, batch_size=EMB_BATCH, shuffle=False, num_workers=0)

print("학습 데이터 임베딩 추출 중...")
all_embs, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        out = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids,
                    output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :].cpu()
        all_embs.append(cls)
        all_labels.append(batch['label'])
        if (i + 1) % 50 == 0:
            print(f"\r  {min((i+1)*EMB_BATCH, len(train_df))}/{len(train_df)}", end='', flush=True)
print(f"\r  {len(train_df)}/{len(train_df)} 완료")

train_embs   = F.normalize(torch.cat(all_embs, dim=0).float(), dim=1)  # (N, D)
train_labels = torch.cat(all_labels, dim=0)                             # (N,)
all_texts    = train_df['메시지내용'].tolist()

out_lines = []

for name, text in QUERIES:
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    if 'token_type_ids' not in enc:
        enc['token_type_ids'] = torch.zeros_like(enc['input_ids'])

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()
        cls_emb = out.hidden_states[-1][:, 0, :].squeeze(0).cpu()

    pred = int(probs.argmax())
    q = F.normalize(cls_emb.float().unsqueeze(0), dim=1)

    sim  = q @ train_embs.T
    dist = (1.0 - sim).clamp(0.0, 2.0)[0]
    topk_dists, topk_idx = dist.topk(10, largest=False)

    # kNN 다수결
    knn_labels = train_labels[topk_idx]
    knn_vote   = knn_labels.mode().values.item()

    out_lines.append("=" * 80)
    out_lines.append(f"[ {name} ]")
    out_lines.append(f"  쿼리: {text}")
    out_lines.append(f"  모델 예측: L{pred} ({probs[pred]*100:.1f}%)  |  kNN 다수결: L{knn_vote}")
    out_lines.append(f"  kNN 분포: " + "  ".join(
        f"L{l}:{(knn_labels == l).sum().item()}개" for l in range(5)
        if (knn_labels == l).sum().item() > 0
    ))
    out_lines.append("")
    out_lines.append("  k   거리     라벨  학습 텍스트")
    out_lines.append("  " + "-" * 74)
    for rank, (d, idx) in enumerate(zip(topk_dists.tolist(), topk_idx.tolist()), 1):
        lbl = train_labels[idx].item()
        neighbor_text = str(all_texts[idx])[:70]
        out_lines.append(f"  {rank:2d}  {d:.5f}  L{lbl}    {neighbor_text}")
    out_lines.append("")

output = "\n".join(out_lines)
out_path = os.path.join(ROOT, '실험', 'knn_neighbors_v19.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"\n-> {out_path} 저장 완료\n")
for name, text in QUERIES:
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    if 'token_type_ids' not in enc:
        enc['token_type_ids'] = torch.zeros_like(enc['input_ids'])
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()
    pred = int(probs.argmax())
    print(f"  {name}: -> L{pred} ({probs[pred]*100:.1f}%)")
