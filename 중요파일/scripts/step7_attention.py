"""
Step 7: Attention 시각화 + 모델 해석

[목적]
- 모델이 어떤 단어/구문에 집중해서 긴급/주의/일반을 판단하는지 시각화
- 설계한 분류 기준이 실제 학습됐는지 검증
- 오분류 케이스에서 모델이 어디를 잘못 봤는지 분석

[방법]
- 마지막 4개 레이어의 어텐션을 평균 → [CLS] 토큰의 각 입력 토큰에 대한 가중치
- 서브워드 토큰 → 원어절 단위로 집계
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def log(msg): print(msg, flush=True)

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MODEL_DIR   = "koelectra_v2_model"
MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
LABEL_COLORS = {'긴급': '#e74c3c', '주의': '#f39c12', '일반': '#3498db'}
LAST_N_LAYERS = 4   # 마지막 N개 레이어 어텐션 평균

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"장치: {device}")

# ─────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────
log("모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, output_attentions=True)
model.to(device).eval()
log("완료")

# ─────────────────────────────────────────
# 어텐션 추출 함수
# ─────────────────────────────────────────
def get_attention(text: str):
    """
    반환: tokens(list), attn_weights(np.array), probs(np.array), pred(int)
    attn_weights: 서브워드 기준 각 토큰이 받는 평균 어텐션 가중치
    """
    enc = tokenizer(text, return_tensors='pt', truncation=True,
                    max_length=MAX_LENGTH, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    probs = F.softmax(out.logits, dim=-1)[0].cpu().numpy()
    pred  = probs.argmax()

    # attentions: tuple[num_layers] of (1, heads, seq, seq)
    # 마지막 LAST_N_LAYERS 레이어 평균, CLS(0번) 토큰의 어텐션
    attn_layers = out.attentions[-LAST_N_LAYERS:]
    # (layers, heads, seq, seq) → 평균 → (seq, seq)
    attn_mean = torch.stack([a[0].mean(dim=0) for a in attn_layers]).mean(dim=0)
    cls_attn  = attn_mean[0].cpu().numpy()   # [CLS]→모든 토큰

    token_ids = enc['input_ids'][0].cpu().numpy()
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)

    # [CLS], [SEP] 제외
    valid_mask  = np.array([t not in ('[CLS]', '[SEP]', '<s>', '</s>', '<pad>')
                             for t in tokens])
    tokens_v    = [t for t, m in zip(tokens, valid_mask) if m]
    attn_v      = cls_attn[valid_mask]
    attn_v      = attn_v / (attn_v.sum() + 1e-9)   # 정규화

    return tokens_v, attn_v, probs, int(pred)


def aggregate_to_words(tokens, attn_weights):
    """
    서브워드(##으로 시작) 토큰을 원어절 단위로 합침 (어텐션은 합산)
    """
    words, word_attns = [], []
    cur_word, cur_attn = '', 0.0
    for tok, w in zip(tokens, attn_weights):
        clean = tok.replace('##', '').replace('▁', '')
        if tok.startswith('##') or tok.startswith('▁') is False and cur_word:
            if tok.startswith('##'):
                cur_word  += clean
                cur_attn  += w
                continue
        if cur_word:
            words.append(cur_word)
            word_attns.append(cur_attn)
        cur_word, cur_attn = clean, w
    if cur_word:
        words.append(cur_word)
        word_attns.append(cur_attn)

    attn_arr = np.array(word_attns)
    if attn_arr.sum() > 0:
        attn_arr = attn_arr / attn_arr.sum()
    return words, attn_arr


# ─────────────────────────────────────────
# 시각화 함수
# ─────────────────────────────────────────
def plot_attention(ax, text, true_label, words, attn, probs, pred,
                   title_extra=''):
    """단일 예시의 어텐션 바 차트"""
    # 상위 15개 토큰만 표시
    n = min(15, len(words))
    idx = np.argsort(attn)[-n:][::-1]
    top_words = [words[i] for i in idx]
    top_attn  = [attn[i]  for i in idx]

    bar_colors = [LABEL_COLORS[LABEL_NAMES[pred]]] * n
    # 어텐션 > 평균+std인 토큰 강조
    threshold = np.mean(top_attn) + np.std(top_attn)
    bar_colors = ['#c0392b' if a >= threshold else LABEL_COLORS[LABEL_NAMES[pred]]
                  for a in top_attn]

    bars = ax.barh(range(n), top_attn, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_words, fontsize=9)
    ax.set_xlabel('어텐션 가중치', fontsize=8)

    pred_str = LABEL_NAMES[pred]
    true_str = LABEL_NAMES[true_label] if true_label is not None else '?'
    correct  = (true_label == pred) if true_label is not None else None

    title = f"실제: {true_str}  |  예측: {pred_str}"
    if correct is False:
        title += "  [오분류]"
    title += f"\n긴급 {probs[0]*100:.1f}%  주의 {probs[1]*100:.1f}%  일반 {probs[2]*100:.1f}%"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=8,
                 color='#c0392b' if correct is False else 'black')

    # 텍스트 맨 아래 표시 (잘림 방지)
    ax.set_xlim(0, max(top_attn) * 1.3)
    ax.invert_yaxis()


# ─────────────────────────────────────────
# 예시 선정
# ─────────────────────────────────────────
EXAMPLES = [
    # (텍스트, 실제레이블, 설명)
    # 긴급 - 명확
    ("[창녕군청] 현재 창녕군 영산면 산불확산 우려, 인근 주민들은 안전한 곳으로 즉시 대피바랍니다.",
     0, "긴급 명확: '즉시 대피' 키워드"),

    ("[행정안전부] 경북 포항 북구 규모 5.4 지진 발생. 여진 대비 안전한 공간으로 이동하세요.",
     0, "긴급 명확: '지진 발생'"),

    # 주의 - 명확
    ("[행정안전부] 오늘 경남(거제) 호우경보 발효. 산사태·침수 위험지역 외출 자제 바랍니다.",
     1, "주의 명확: '호우경보 발효'"),

    ("[기상청] 내일 오전까지 수도권 대설주의보 발효. 빙판길 안전운전 바랍니다.",
     1, "주의 명확: '대설주의보 발효'"),

    # 일반 - 명확
    ("[전북경찰청] 전주시에서 실종된 민도식씨(남,70세)를 찾습니다-160cm,회색줄무늬티,검정바지",
     2, "일반 명확: 실종자 수색"),

    ("[세종시청] 조치원읍 신안리 일원 상수도관 누수로 단수 중, 12시경 복구 완료 예정입니다.",
     2, "일반 명확: 단수 안내"),

    # 경계 케이스
    ("[보령시] 오천면 일원 산사태 주의보 발령. 산간지역 주민께서는 마을회관으로 대피를 권고드립니다.",
     1, "경계: 대피 권고(긴급↔주의)"),

    ("[합천군] 오늘 22:10경 황강 수위 급격히 상승, 범람이 우려됩니다. 인근 주민은 상황 예의주시 바랍니다.",
     1, "경계: 범람 우려(주의↔긴급)"),

    # 오분류 가능성 케이스
    ("[나주시청] 확진자 2명 발생. 동선 및 접촉자 파악 완료. 방역수칙 준수 바랍니다.",
     2, "모호: 확진자 발생(일반↔주의)"),

    ("[남양주시] 북한강 수위 상승으로 하천변 주민은 즉시 대피하십시오.",
     0, "긴급: '즉시 대피' + 범람"),
]

# ─────────────────────────────────────────
# 메인 시각화
# ─────────────────────────────────────────
log(f"\n{len(EXAMPLES)}개 예시 어텐션 분석 중...")

fig, axes = plt.subplots(2, 5, figsize=(22, 14))
fig.suptitle('KoELECTRA-v2 Attention 시각화\n(모델이 집중한 상위 15개 토큰)', fontsize=14)
axes = axes.flatten()

results = []
for i, (text, true_label, desc) in enumerate(EXAMPLES):
    tokens, attn, probs, pred = get_attention(text)
    words, word_attn = aggregate_to_words(tokens, attn)
    plot_attention(axes[i], text, true_label, words, word_attn, probs, pred, desc)
    correct = (pred == true_label)
    results.append({
        'text': text[:60] + '...',
        'true': LABEL_NAMES[true_label],
        'pred': LABEL_NAMES[pred],
        'correct': correct,
        'top_token': words[np.argmax(word_attn)] if words else '',
        '긴급%': round(probs[0]*100, 1),
        '주의%': round(probs[1]*100, 1),
        '일반%': round(probs[2]*100, 1),
    })
    log(f"  [{i+1}] {LABEL_NAMES[true_label]}→{LABEL_NAMES[pred]} "
        f"{'O' if correct else 'X'}  "
        f"top_token: '{words[np.argmax(word_attn)] if words else '?'}'  {desc}")

plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
log("저장: attention_visualization.png")

# ─────────────────────────────────────────
# 클래스별 고어텐션 토큰 통계
# ─────────────────────────────────────────
log("\n클래스별 고어텐션 토큰 통계 분석 중...")
test_df = pd.read_csv('data_test.csv', encoding='utf-8-sig').fillna({'메시지내용': ''})

# 클래스별 50개 샘플에서 top-3 토큰 수집
from collections import Counter
class_top_tokens = {0: Counter(), 1: Counter(), 2: Counter()}

for lbl in range(3):
    samples = test_df[test_df['label'] == lbl].sample(
        min(50, len(test_df[test_df['label'] == lbl])), random_state=42)
    for text in samples['메시지내용']:
        try:
            tokens, attn, probs, pred = get_attention(text)
            words, word_attn = aggregate_to_words(tokens, attn)
            if len(words) == 0:
                continue
            top3_idx = np.argsort(word_attn)[-3:]
            for idx in top3_idx:
                tok = words[idx].strip()
                if len(tok) >= 2:
                    class_top_tokens[lbl][tok] += 1
        except Exception:
            continue
    log(f"  {LABEL_NAMES[lbl]} 완료")

# 클래스별 상위 토큰 바 차트
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('클래스별 고어텐션 상위 토큰 (50개 샘플 기준)', fontsize=13)

for i, (lbl, counter) in enumerate(class_top_tokens.items()):
    top15 = counter.most_common(15)
    if not top15:
        continue
    words_c, counts = zip(*top15)
    axes[i].barh(range(len(words_c)), counts,
                 color=LABEL_COLORS[LABEL_NAMES[lbl]], edgecolor='white')
    axes[i].set_yticks(range(len(words_c)))
    axes[i].set_yticklabels(words_c, fontsize=9)
    axes[i].set_title(f'{LABEL_NAMES[lbl]} 클래스', fontsize=11)
    axes[i].set_xlabel('등장 빈도')
    axes[i].invert_yaxis()

plt.tight_layout()
plt.savefig('attention_top_tokens.png', dpi=150, bbox_inches='tight')
log("저장: attention_top_tokens.png")

# ─────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────
results_df = pd.DataFrame(results)
log("\n=== 예시 분석 결과 ===")
log(results_df[['true', 'pred', 'correct', 'top_token', '긴급%', '주의%', '일반%']].to_string())

with open('attention_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Attention 분석 결과 ===\n\n")
    f.write(results_df.to_string())
    f.write("\n\n=== 클래스별 고어텐션 토큰 Top 15 ===\n")
    for lbl in range(3):
        f.write(f"\n[{LABEL_NAMES[lbl]}]\n")
        for tok, cnt in class_top_tokens[lbl].most_common(15):
            f.write(f"  {tok}: {cnt}회\n")
log("저장: attention_summary.txt")

log("\n=== Step 7 완료 ===")
