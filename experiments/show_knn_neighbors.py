"""
특정 문자의 kNN 이웃 10개를 학습 데이터에서 직접 꺼내서 보여준다.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_v2 import load_and_split_v2
from ood_knn import KNNOOD

MODEL_DIR = os.path.join(ROOT, 'model_v9n')
TOK_DIR   = os.path.join(ROOT, 'tokenizer_v9n')
DATA_PATH = os.path.join(ROOT, '중요파일', 'data', 'raw',
                         '재난문자_레이블링결과_dedup_v2.xlsx')
OOD_STATS = os.path.join(ROOT, '실험', 'knn_ood_stats.pt')
MAX_LEN   = 96

QUERIES = [
    ("바이러스변종 (정답 L2, 예측 L0)",
     "바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고."),
    ("신종호흡기 (정답 L1, 예측 L0)",
     "신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화."),
]

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("데이터 로드 중...")
train_df, _, _ = load_and_split_v2(DATA_PATH)
all_texts = train_df['text'].tolist()

print("모델 / kNN 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

detector = KNNOOD.load(OOD_STATS)
train_embs = detector.train_embs      # (N, D) already L2-normalized
train_labels = detector.train_labels  # (N,)

out_lines = []

for name, text in QUERIES:
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()
        cls_emb = out.hidden_states[-1][:, 0, :].squeeze(0).cpu()

    pred = int(probs.argmax())
    q = F.normalize(cls_emb.float().unsqueeze(0), dim=1)

    sim  = q @ train_embs.T
    dist = (1.0 - sim).clamp(0.0, 2.0)[0]
    topk_dists, topk_idx = dist.topk(10, largest=False)

    out_lines.append("=" * 80)
    out_lines.append(f"[ {name} ]")
    out_lines.append(f"  쿼리: {text}")
    out_lines.append(f"  모델 예측: {LABEL_NAMES[pred]} ({probs[pred]*100:.1f}%)")
    out_lines.append(f"  kNN 다수결: L{detector.train_labels[topk_idx].mode().values.item()}")
    out_lines.append("")
    out_lines.append("  k   거리     라벨  학습 텍스트")
    out_lines.append("  " + "-" * 74)
    for rank, (d, idx) in enumerate(zip(topk_dists.tolist(), topk_idx.tolist()), 1):
        lbl = train_labels[idx].item()
        neighbor_text = all_texts[idx][:70]
        out_lines.append(f"  {rank:2d}  {d:.5f}  L{lbl}    {neighbor_text}")
    out_lines.append("")

output = "\n".join(out_lines)

out_path = os.path.join(ROOT, '실험', 'knn_neighbors.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"-> {out_path} 저장 완료")
print("(터미널 인코딩 문제로 내용은 파일에서 확인하세요)")
